from __future__ import print_function
__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from ..NSfracStep import *
from numpy import pi, arctan, array
import sys
set_log_level(99)

# Override some problem specific parameters
def problem_parameters(NS_parameters, NS_expressions, commandline_kwargs, **NS_namespace):
    eps  = 1e-6
    #ux_mms = "cos(x[0] + x[1]) * t_e + eps"
    #uy_mms = "-cos(x[0] + x[1]) * t_e + eps"
    #p_mms = "(sin(x[1]) + cos(x[0])) * t_e + eps"
    #ux_mms = "(-sin(x[1] * pi) + cos(x[0] * pi)) * (t_e + 1) + eps"
    #uy_mms = "(-pi*sin(x[0] * pi)*x[1] + cos(x[0] * pi)) * (t_e + 1) + eps"
    #p_mms  = "cos(x[1] * pi) * sin(x[0] * pi) * (t_e + 1) + eps"
    ux_mms = " pi*cos(t_e) * sin(2*pi*x[1]) * sin(pi*x[0])*sin(pi*x[0]) + eps "
    uy_mms = "-pi*cos(t_e) * sin(2*pi*x[0]) * sin(pi*x[1])*sin(pi*x[1]) + eps "
    p_mms  = " -cos(t_e) * cos(pi*x[0]) * sin(pi*x[1]) + eps "

    NS_parameters.update(dict(
        ux_mms = ux_mms,
        uy_mms = uy_mms,
        p_mms  = p_mms,
        nu   = 1,
        dt   = 0.001,
        T    = 0.005,
        N    = 40,
        eps  = eps,
        print_intermediate_info = 1e10,
        folder="MMS_results",
        iters_on_first_timestep = 1,
        max_iter  = 1,
        max_error = 1e-9,
        velocity_degree = 1,
        pressure_degree = 1,
        use_krylov_solvers = False,
        print_velocity_pressure_convergence = False))

    if "velocity_degree" in commandline_kwargs.keys():
        v_degree = commandline_kwargs["velocity_degree"]
    else:
        v_degree = NS_parameters["velocity_degree"]

    if "pressure_degree" in commandline_kwargs.keys():
        p_degree = commandline_kwargs["pressure_degree"]
    else:
        p_degree = NS_parameters["pressure_degree"]

    NS_expressions.update(
        ux_e = Expression(ux_mms, eps=eps, degree=v_degree, t_e=0),
        uy_e = Expression(uy_mms, eps=eps, degree=v_degree, t_e=0),
        p_e  = Expression(p_mms, eps=eps, degree=p_degree, t_e=0),
        t_e  = Constant(0.0) )


# Create a mesh here
def mesh(N, **params):
    #m = UnitSquareMesh(N, N)
    m = RectangleMesh(Point(0.0, 0.0), Point(1.0, 1.0), N, N)
    return m


def create_bcs(ux_e, uy_e, p_e, V, Q, sys_comp, mesh, **NS_namespace):
    external = AutoSubDomain(lambda x, b: b)
    boundary = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundary.set_all(0)
    external.mark(boundary, 1)

    bcs = dict((ui, []) for ui in sys_comp)
    bc0 = DirichletBC(V, ux_e, boundary, 1)
    bc1 = DirichletBC(V, uy_e, boundary, 1)
    bcp = DirichletBC(Q, p_e, boundary, 1)
    bcs['u0'] = [bc0]
    bcs['u1'] = [bc1]
    bcs['p']  = [bcp]

    return bcs


def body_force(V, Q, mesh, ux_mms, uy_mms, p_mms, nu, t_e, eps, **NS_namespace):
    x = SpatialCoordinate(mesh)
    u_vec = as_vector([eval(ux_mms), eval(uy_mms)])
    p_    = eval(p_mms)

    f     = ( diff(u_vec, t_e)
            + dot(u_vec,nabla_grad(u_vec))
            + div(p_ * Identity(2))
            - nu*div((grad(u_vec) + grad(u_vec).T)))

    return f


def initialize(V, Q, q_1, q_2, ux_e, uy_e, p_e, dt, **NS_namespace):
    ux_e.t_e = -dt
    uy_e.t_e = -dt

    ux = project(ux_e, V)
    uy = project(uy_e, V)

    q_2["u0"].vector().axpy(1, ux.vector())
    q_2["u1"].vector().axpy(1, uy.vector())

    if 'IPCS' in NS_parameters['solver']:
        p_e.t_e  = 0.5*dt
    else:
        p_e.t_e  = 0
    ux_e.t_e = 0
    uy_e.t_e = 0

    ux = project(ux_e, V)
    uy = project(uy_e, V)
    p_ = project(p_e, Q)

    q_1["u0"].vector().axpy(1, ux.vector())
    q_1["u1"].vector().axpy(1, uy.vector())
    q_1["p"].vector().axpy(1, p_.vector())


def pre_solve_hook(V, Q, mesh, newfolder, q_, t, velocity_degree, **NS_namespace):
    """Called prior to time loop"""
    viz_sol = XDMFFile(MPI.comm_world, path.join(newfolder, "VTK","solution.xdmf"))
    viz_sol.parameters["rewrite_function_mesh"] = True
    viz_sol.parameters["flush_output"] = True
    pmms    = Function(Q, name="p_mms")
    uxmms   = Function(V, name="ux_mms")
    uymms   = Function(V, name="uy_mms")
    p_sol    = Function(Q, name="p_sol")
    ux_sol   = Function(V, name="ux_sol")
    uy_sol   = Function(V, name="uy_sol")
    p_error  = Function(Q, name="p_error")
    ux_error = Function(V, name="ux_error")
    uy_error = Function(V, name="uy_error")

    return dict(viz_sol=viz_sol, p_sol=p_sol, ux_sol=ux_sol, uy_sol=uy_sol,
                p_error=p_error, ux_error=ux_error, uy_error=uy_error,
                pmms=pmms, uxmms=uxmms, uymms=uymms)


def start_timestep_hook(t, dt, p_e, ux_e, uy_e, t_e, **NS_namespace):
    if 'IPCS' in NS_parameters['solver']:
        p_e.t_e  = t - 0.5*dt
    else:
        p_e.t_e  = t
    ux_e.t_e = t
    uy_e.t_e = t
    t_e.assign(t)


def temporal_hook(t, dt, q_, viz_sol, p_e, p_error, pmms, p_sol, ux_e, ux_error,
                  uxmms, ux_sol, uy_e, uy_error, uymms, uy_sol, V, Q, **NS_namespace):

    p = interpolate(p_e, Q)
    p_sol.vector().zero()
    p_sol.vector().axpy(1, q_["p"].vector())
    pmms.vector().zero()
    pmms.vector().axpy(1, p.vector())
    p_error.vector().zero()
    p_error.vector().axpy(1, p.vector())
    p_error.vector().axpy(-1, q_["p"].vector())

    ux = interpolate(ux_e, V)
    ux_sol.vector().zero()
    ux_sol.vector().axpy(1, q_["u0"].vector())
    uxmms.vector().zero()
    uxmms.vector().axpy(1, ux.vector())
    ux_error.vector().zero()
    ux_error.vector().axpy(1, ux.vector())
    ux_error.vector().axpy(-1, q_["u0"].vector())

    uy = interpolate(uy_e, V)
    uy_sol.vector().zero()
    uy_sol.vector().axpy(1, q_["u1"].vector())
    uymms.vector().zero()
    uymms.vector().axpy(1, uy.vector())
    uy_error.vector().zero()
    uy_error.vector().axpy(1, uy.vector())
    uy_error.vector().axpy(-1, q_["u1"].vector())

    viz_sol.write(ux_sol, t)
    viz_sol.write(uxmms, t)
    viz_sol.write(ux_error, t)
    viz_sol.write(uy_sol, t)
    viz_sol.write(uymms, t)
    viz_sol.write(uy_error, t)
    viz_sol.write(p_sol, t)
    viz_sol.write(pmms, t)
    viz_sol.write(p_error, t)

# Print error at the end of the computation
def theend_hook(V, Q, ux_e, uy_e, q_, T, dt, mesh, p_e, **NS_namespace):
    ux = interpolate(ux_e, V)
    uy = interpolate(uy_e, V)
    p  = interpolate(p_e, Q)
    print(" ")
    print("Scheme {}".format(NS_parameters['solver']))
    print("T      {0:.6e}".format(T))
    print("dt     {0:.6e}".format(dt))
    print("dx     {0:.6e}".format(mesh.hmin()))
    print("L2 norm (ux-uxmms) {0:.6e}".format(errornorm(ux, q_["u0"], norm_type="l2",degree_rise=3)))
    print("L2 norm (uy-uymms) {0:.6e}".format(errornorm(uy, q_["u1"], norm_type="l2",degree_rise=3)))
    print("L2 norm (p-pmms)   {0:.6e}".format(errornorm(p, q_["p"], norm_type="l2", degree_rise=3)))
    print(" ")
