from ..NSfracStep import *
import sys
import pickle
from os import makedirs
import random
import numpy as np
from mshr import *
import matplotlib.pyplot as plt

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
        T = 1
        dt = 0.01
        nu = 10
        NS_parameters.update(
            checkpoint = 1000,
            save_step = 10,
            check_flux = 2,
            print_intermediate_info = 1000,
            nu = nu,
            T  = T,
            dt = dt,
            N  = 50,
            R1 = 5,
            R0 = 1,
            #mesh_path = "mesh/box.xml",
            velocity_degree = 2,
            pressure_degree = 1,
            folder = "prescribed_circle_results",
            use_krylov_solvers = True,
            max_iter = 5 # number of velocity correction iterations
            )


def mesh(R1, R0, N, **NS_namespace):
    domain = Circle(Point(0.0,0.0), R1) - Circle(Point(0.0,0.0), R0)
    mesh   = generate_mesh(domain, N)
    #mesh   = refine(mesh)
    #plot(mesh)
    #plt.show()
    #import pdb; pdb.set_trace()
    return mesh


class Walls(UserExpression):
    def __init__(self, comp, w_, **kwargs):
        self.w_ = w_["u%d" % comp]
        super().__init__(**kwargs)
    def eval(self, value, x):
        value[:] = [self.w_(x)]


def create_bcs(V, Q, w_, sys_comp, u_components, mesh, newfolder, NS_expressions, R1, R0, **NS_namespace):
    info_red("Creating boundary conditions")

    # External and inner cercle boundaries
    wall_ext = AutoSubDomain(lambda x, b: b and (sqrt(x[0]**2.0 + x[1]**2.0) > (R0*1.1)))
    wall_int = AutoSubDomain(lambda x, b: b and (sqrt(x[0]**2.0 + x[1]**2.0) < (R0*1.1)))
    boundary = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundary.set_all(0)
    wall_ext.mark(boundary, 1)
    wall_int.mark(boundary, 2)
    f = File("test_b.pvd")
    f<<boundary
    #from IPython import embed; embed()
    walls0 = Walls(0, w_, element=V.ufl_element())
    walls1 = Walls(1, w_, element=V.ufl_element())
    NS_expressions["walls0"] = walls0
    NS_expressions["walls1"] = walls1

    # Boundary condition values
    bcs  = dict((ui, []) for ui in sys_comp)
    bcu0 = DirichletBC(V, walls0, boundary, 1)
    bcu1 = DirichletBC(V, walls1, boundary, 1)
    bcp  = DirichletBC(Q, Constant(0), boundary, 2)
    bcs['u0'] = [bcu0]
    bcs['u1'] = [bcu1]
    bcs['p']  = [bcp]

    return bcs


def pre_solve_hook(W, V, u_, mesh, newfolder, T, d_, velocity_degree, **NS_namespace):
    """Called prior to time loop"""
    viz_d = XDMFFile(MPI.comm_world, path.join(newfolder, "VTK", "deformation.xdmf"))
    viz_u = XDMFFile(MPI.comm_world, path.join(newfolder, "VTK", "velocity.xdmf"))
    viz_p = XDMFFile(MPI.comm_world, path.join(newfolder, "VTK", "pressure.xdmf"))
    viz_w = XDMFFile(MPI.comm_world, path.join(newfolder, "VTK", "mesh_velocity.xdmf"))

    for viz in [viz_d, viz_u, viz_p, viz_w]:
        viz.parameters["rewrite_function_mesh"] = True
        viz.parameters["flush_output"] = True

    Vv = VectorFunctionSpace(mesh, "CG", velocity_degree)
    u_vec = Function(Vv, name="u")

    # Set up mesh solver
    u_mesh, v_mesh = TrialFunction(W), TestFunction(W)
    f_mesh = Function(W)

    F_mesh = inner(grad(u_mesh), grad(v_mesh))*dx + inner(f_mesh, v_mesh)*dx
    a_mesh = lhs(F_mesh)
    l_mesh = rhs(F_mesh)

    A_mesh = assemble(a_mesh)
    L_mesh = assemble(l_mesh)

    wall_ext = AutoSubDomain(lambda x, b: b and (sqrt(x[0]**2.0 + x[1]**2.0) > 1.1))
    wall_int = AutoSubDomain(lambda x, b: b and (sqrt(x[0]**2.0 + x[1]**2.0) < 1.1))

    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    wall_ext.mark(boundaries, 1)
    wall_int.mark(boundaries, 2)

    wall_ext_exp = Expression(("-x[0]*0.015", "-x[1]*0.015"), degree=velocity_degree)
    #wall_ext_ex = Expression(("-x[0]*0.1*t", "-x[1]*0.1*t"), t=0, degree=1)
    wall_int_exp = Constant((0, 0))

    wall_ext_bc = DirichletBC(W, wall_ext_exp, boundaries, 1)
    wall_int_bc = DirichletBC(W, wall_int_exp, boundaries, 2)

    bc_mesh = [wall_ext_bc, wall_int_bc]

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
                F_mesh=F_mesh, bc_mesh=bc_mesh, w_vec=w_vec, viz_w=viz_w,
                a_mesh=a_mesh, l_mesh=l_mesh, A_mesh=A_mesh, L_mesh=L_mesh, boundaries=boundaries)


def update_prescribed_motion(t, dt, d_, d_1, w_, u_components, tstep, mesh_sol, F_mesh,
                             bc_mesh, w_vec, NS_expressions,
                             a_mesh, l_mesh, A_mesh, L_mesh,
                             **NS_namespace):

    # Solve for d and w
    assemble(a_mesh, tensor=A_mesh)
    assemble(l_mesh, tensor=L_mesh)

    for bc in bc_mesh:
        bc.apply(A_mesh, L_mesh)

    mesh_sol.solve(A_mesh, d_.vector(), L_mesh)

    w_vec.vector().zero()
    w_vec.vector().axpy(1/dt, d_.vector())

    # Read velocity
    for i, ui in enumerate(u_components):
        assign(w_[ui], w_vec.sub(i))


def temporal_hook(t, d_, w_, q_, f, tstep, viz_u, viz_d, viz_p, u_components,
                  u_vec, viz_w, mesh, R0, boundaries, **NS_namespace):
    assign(u_vec.sub(0), q_["u0"])
    assign(u_vec.sub(1), q_["u1"])

    viz_d.write(d_, t)
    viz_u.write(u_vec, t)
    viz_p.write(q_['p'], t)
    viz_w.write(w_['u0'], t)
    viz_w.write(w_['u1'], t)

    # Compute the fluid flux at the inner boundary
    ds    = Measure("ds", subdomain_data = boundaries)
    n     = FacetNormal(mesh)
    Q_ext = dot(u_vec,n)*ds(1)
    Q_int = dot(u_vec,n)*ds(2)

    Flux_int  = assemble(Q_int)
    Flux_ext  = assemble(Q_ext)
    Flux_diff = Flux_int+Flux_ext

    if tstep % 10 == 0:
        print("Flux_int: %e" % Flux_int, "   Flux_ext: %e" % Flux_ext, "   Difference: %e" % Flux_diff)
        print("Time:", round(t,3))
