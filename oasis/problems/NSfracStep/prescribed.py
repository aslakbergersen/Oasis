from ..NSfracStep import *
import sys
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
        T = 2.0
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
            Lx = 5,
            Ly = 1,
            #mesh_path = "mesh/box.xml",
            velocity_degree = 1,
            folder = "prescribed_results",
            use_krylov_solvers = True,
            )


def mesh(Lx, Ly, N, **NS_namespace):
    mesh = RectangleMesh(Point(0.0, 0.0), Point(Lx, Ly), N*Lx, N*Ly*2)
    return mesh


class Walls(UserExpression):
    def __init__(self, comp, w_, **kwargs):
        self.w_ = w_["u%d" % comp]
        super().__init__(**kwargs)

    def eval(self, value, x):
        value[:] = [self.w_(x)]


def create_bcs(V, Q, w_, sys_comp, u_components, mesh, newfolder, NS_expressions, Lx, Ly, **NS_namespace):
    info_red("Creating boundary conditions")

    outlet = AutoSubDomain(lambda x, b: b and near(x[0], Lx))
    walls = AutoSubDomain(lambda x, b: b and (near(x[0], 0) or
                                              near(x[1], 0) or
                                              near(x[1], Ly)))
    boundary = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundary.set_all(0)
    walls.mark(boundary, 1)
    outlet.mark(boundary, 2)

    walls0 = Walls(0, w_, element=V.ufl_element())
    walls1 = Walls(1, w_, element=V.ufl_element())
    NS_expressions["walls0"] = walls0
    NS_expressions["walls1"] = walls1

    bcs = dict((ui, []) for ui in sys_comp)
    bcu0 = DirichletBC(V, walls0, boundary, 1)
    bcu1 = DirichletBC(V, walls1, boundary, 1)
    bcp = DirichletBC(Q, Constant(5), boundary, 2)
    bcs['u0'] = [bcu0]
    bcs['u1'] = [bcu1]
    bcs["p"] = [bcp]

    return bcs


def pre_solve_hook(W, V, u_, mesh, newfolder, T, d_, **NS_namespace):
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
    u_mesh, v_mesh = TrialFunction(W), TestFunction(W)
    f_mesh = Function(W)

    F_mesh = inner(grad(u_mesh), grad(v_mesh))*dx + inner(f_mesh, v_mesh)*dx
    a_mesh = lhs(F_mesh)
    l_mesh = rhs(F_mesh)

    A_mesh = assemble(a_mesh)
    L_mesh = assemble(l_mesh)

    left = AutoSubDomain(lambda x, b: b and near(x[0], 0))
    rigid = AutoSubDomain(lambda x, b: x[0] >= 3)

    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    left.mark(boundaries, 1)
    rigid.mark(boundaries, 2)

    left_ex = Expression(("t", "0"), omega=T, n=2, t=0, degree=1)
    rigid_ex = Constant((0, 0))

    left_bc = DirichletBC(W, left_ex, boundaries, 1)
    rigid_bc = DirichletBC(W, rigid_ex, boundaries, 2)

    bc_mesh = [left_bc, rigid_bc]

    mesh_prec = PETScPreconditioner("ilu")  # in tests sor are faster .. 
    mesh_sol = PETScKrylovSolver("gmres", mesh_prec)
    w_vec = Function(Vv)

    krylov_solvers = dict(monitor_convergence=False,
                          report=False,
                          error_on_nonconvergence=False,
                          nonzero_initial_guess=True,
                          maximum_iterations=200,
                          relative_tolerance=1e-10,
                          absolute_tolerance=1e-10)

    mesh_sol.parameters.update(krylov_solvers)

    return dict(viz_p=viz_p, viz_u=viz_u, viz_d=viz_d, u_vec=u_vec, mesh_sol=mesh_sol,
                left_ex=left_ex, F_mesh=F_mesh, bc_mesh=bc_mesh, w_vec=w_vec, viz_w=viz_w,
                a_mesh=a_mesh, l_mesh=l_mesh, A_mesh=A_mesh, L_mesh=L_mesh)


def update_prescribed_motion(t, dt, d_, d_1, w_, u_components, tstep, mesh_sol, F_mesh,
                             bc_mesh, w_vec, left_ex, NS_expressions,
                             a_mesh, l_mesh, A_mesh, L_mesh,
                             **NS_namespace):
    # Update time
    left_ex.t = t

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
