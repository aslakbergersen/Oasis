from ..NSfracStep import *
import pickle
from os import makedirs
import random
import numpy as np

set_log_level(99)
parameters["allow_extrapolation"] = True

def problem_parameters(commandline_kwargs, NS_parameters, NS_expressions, **NS_namespace):
    if "restart_folder" in commandline_kwargs.keys():
        restart_folder = commandline_kwargs["restart_folder"]
        restart_folder = path.join(getcwd(), restart_folder)

        f = open(path.join(path.dirname(path.abspath(__file__)), restart_folder, 'params.dat'), 'r')
        NS_parameters.update(cPickle.load(f))
        NS_parameters['T'] = 10
        NS_parameters['restart_folder'] = restart_folder
        globals().update(NS_parameters)

    else:
        T = 2.
        dt = 0.01
        nu = 10
        NS_parameters.update(
            checkpoint = 1000,
            save_step = 10,
            check_flux = 2,
            print_intermediate_info = 1000,
            nu = nu,
            T = T,
            dt = dt,
            N = 20,
            #mesh_path = "mesh/box.xml",
            velocity_degree = 1,
            folder = "prescribed_results",
            use_krylov_solvers = True,
            )


def mesh(N, **NS_namespace):
    mesh = RectangleMesh(Point(0.0, 0.0), Point(3.0, 1.0), N*3, N)
    return mesh

# User expression
class test(UserExpression):
    def __init__(self, dummy, dummy1, **kwargs):
        self.dummy = dummy
        self.dummy1 = dummy1
        super().__init__(**kwargs)

    def eval(self, value, x):
        value[:] = x[0]


class Walls(UserExpression):
    def __init__(self, comp, w_, **kwargs):
        self.w_ = w_["u%d" % comp]
        super().__init__(**kwargs)

    def eval(self, value, x):
        value[:] = self.w_(x)


def create_bcs(V, Q, w_, sys_comp, u_components, mesh, newfolder, NS_expressions, **NS_namespace):
    # NOTE: This assumes that the w function starts from 0. If not read the
    # initial condition in create_bcs
    info_red("Creating boundary conditions")

    outlet = AutoSubDomain(lambda x, b: b and near(x[0], 3))
    walls = AutoSubDomain(lambda x, b: b and (near(x[0], 0) or
                                              near(x[1], 0) or
                                              near(x[1], 1)))
    boundary = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundary.set_all(0)
    walls.mark(boundary, 1)
    outlet.mark(boundary, 2)

    w_exp = Expression("(1./3)*t*(3 - x[0])", t=0, element=V.ufl_element())
    NS_expressions["w_exp"] = w_exp

    #walls0 = Walls(0, w_, element=V.ufl_element())
    #walls1 = Walls(1, w_, element=V.ufl_element())

    walls0 = test(0, w_, element=V.ufl_element())
    walls1 = test(0, w_, element=V.ufl_element())

    #f = File(path.join(newfolder, "VTK", "boundary.pvd"))
    #f << boundary

    bcs = dict((ui, []) for ui in sys_comp)
    #bcu0 = DirichletBC(V, w_exp, boundary, 1)
    bcu1 = DirichletBC(V, Constant(0), boundary, 1)
    bcu0 = DirichletBC(V, walls0, boundary, 1)
    #bcu1 = DirichletBC(V, walls1, boundary, 1)

    #bcu0 = DirichletBC(V, Constant(0), boundary, 1)
    #bcu1 = DirichletBC(V, Constant(0), boundary, 1) # Should be constant?
    bcp = DirichletBC(Q, Constant(5), boundary, 2)
    bcs['u0'] = [bcu0]
    bcs['u1'] = [bcu1]
    bcs["p"] = [bcp]

    return bcs


def pre_solve_hook(V, u_, mesh, newfolder, T, d_, **NS_namespace):
    """Called prior to time loop"""
    viz_d = XDMFFile(MPI.comm_world, path.join(newfolder, "VTK", "deformation.xdmf"))
    viz_u = XDMFFile(MPI.comm_world, path.join(newfolder, "VTK", "velocity.xdmf"))
    viz_p = XDMFFile(MPI.comm_world, path.join(newfolder, "VTK", "pressure.xdmf"))
    viz_w = XDMFFile(MPI.comm_world, path.join(newfolder, "VTK", "mesh_velocity.xdmf"))

    for viz in [viz_d, viz_u, viz_p, viz_w]:
        viz.parameters["rewrite_function_mesh"] = True
        viz.parameters["flush_output"] = True

    Vv = VectorFunctionSpace(mesh, "CG", 1)
    u_vec = Function(Vv, name="u")

    # Set up mesh solver
    # NOTE: Consider other pre.con. f.ex. jacobi 
    u_mesh, v_mesh = TrialFunction(Vv), TestFunction(Vv)
    f_mesh = Function(Vv)
    #alfa = 1. / det(Identity(len(d_)) + grad(d_))

    F = inner(grad(u_mesh), grad(v_mesh))*dx + inner(f_mesh,v_mesh)*dx
    a_mesh = lhs(F)
    l_mesh = rhs(F)

    A_mesh = assemble(a_mesh)
    L_mesh = assemble(l_mesh)

    left = AutoSubDomain(lambda x, b: b and near(x[0], 0))
    walls = AutoSubDomain(lambda x, b: b and near(x[1], 1) and near(x[1], 0))
    rigid = AutoSubDomain(lambda x, b: 3 <= x[0])

    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    left.mark(boundaries, 1)
    rigid.mark(boundaries, 2)
    walls.mark(boundaries, 3)

    left_ex = Expression(("t", "0"), omega=T, n=2, t=0, degree=2)
    rigid_ex = Constant((0, 0))
    walls_ex =  Expression("(1./3)*t*(3 - x[0])", t=0, degree=2)

    left_bc = DirichletBC(Vv, left_ex, boundaries, 1)
    rigid_bc = DirichletBC(Vv, rigid_ex, boundaries, 2)
    #wall_bc = DirichletBC(Vv, walls_ex, boundaries, 3)

    bc_mesh = [left_bc, rigid_bc] #, wall_bc]

    mesh_prec = PETScPreconditioner("ilu")
    mesh_sol = PETScKrylovSolver("gmres", mesh_prec) #, mesh_prec)
    w_vec = Function(Vv)

    return dict(viz_p=viz_p, viz_u=viz_u, viz_d=viz_d,
                u_vec=u_vec, mesh_sol=mesh_sol, left_ex=left_ex, a_mesh=a_mesh,
                l_mesh=l_mesh, A_mesh=A_mesh, L_mesh=L_mesh, bc_mesh=bc_mesh,
                w_vec=w_vec, viz_w=viz_w)


def update_prescribed_motion(t, dt, d_, d_1, w_, u_components, tstep,
                             mesh_sol, a_mesh, l_mesh, A_mesh, L_mesh, bc_mesh,
                             w_vec, left_ex, NS_expressions, **NS_namespace):
    # Update time
    left_ex.t = t
    NS_expressions["w_exp"].t = t

    # Read deformation
    d_1.vector().zero()
    d_1.vector().axpy(1, d_.vector())

    # Solve for d and w
    assemble(a_mesh, tensor=A_mesh)
    assemble(l_mesh, tensor=L_mesh)

    for bc in bc_mesh:
        bc.apply(A_mesh, L_mesh)

    mesh_sol.solve(A_mesh, d_.vector(), L_mesh)

    w_vec.vector().zero()
    w_vec.vector().axpy(1/dt, d_.vector())
    w_vec.vector().axpy(-1/dt, d_1.vector())

    # Read velocity
    for i, ui in enumerate(u_components):
        assign(w_[ui], w_vec.sub(i))


def temporal_hook(t, d_, w_, q_, f, tstep, viz_u, viz_d, viz_p, u_components,
                  u_vec, viz_w, **NS_namespace):
    assign(u_vec.sub(0), q_["u0"])
    assign(u_vec.sub(1), q_["u1"])

    viz_d.write(d_, t)
    viz_u.write(u_vec, t)
    viz_p.write(q_["p"], t)
    viz_w.write(w_["u0"], t)
    viz_w.write(w_["u1"], t)

    if tstep % 10 == 0:
        print("Time:", round(t,3))
