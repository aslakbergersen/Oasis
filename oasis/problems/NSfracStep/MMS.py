from __future__ import print_function
__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from ..NSfracStep import *
import numpy as np
import sys
set_log_level(99)


# Override some problem specific parameters
def problem_parameters(NS_parameters, NS_expressions, commandline_kwargs, **NS_namespace):
    eps = 0

    # From J. L. Guermond et al. 2005
    ux_mms = " pi*cos(t_e) * sin(2*x[1]) * sin(x[0]) * sin(x[0]) + eps + x[1]/pi"
    uy_mms = "-pi*cos(t_e) * sin(2*x[0]) * sin(x[1]) * sin(x[1]) + eps"
    p_mms = "cos(t_e) * cos(x[0]) * sin(x[1]) + eps "
    #ux_mms_dx = "2*pi*pi * cos(t_e) * cos(2*pi*x[1]) * sin(pi*x[0])*sin(pi*x[0])"
    #uy_mms = "-" + ux_mms_dx + "*x[1] + cos(x[0]*pi) + eps"

    # Our suggestion for a divergence free solution
    #ux_mms = "(cos(x[0]) + sin(x[1])) * exp(t_e) + eps "
    #uy_mms = "(sin(x[0])*x[1] + sin(x[0])) * exp(t_e) + eps "
    #p_mms  = "exp(t_e) * cos(x[0]) * sin(x[1]) + eps"

    # Prescribed motion
    wx_mms = "x[1]/pi" #x[0]" #ux_mms #+ " + cos(t_e) * sin(2*x[0]) * sin(2*x[1])"
    wy_mms = "0" #uy_mms #+ " + cos(t_e) * sin(2*x[0]) * sin(2*x[1])"

    NS_parameters.update(dict(
        ux_mms = ux_mms,
        uy_mms = uy_mms,
        wx_mms = wx_mms,
        wy_mms = wy_mms,
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
        velocity_degree = 2,
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
        wx_e = Expression(wx_mms, eps=eps, degree=v_degree, t_e=0),
        wy_e = Expression(wy_mms, eps=eps, degree=v_degree, t_e=0),
        ux_e = Expression(ux_mms, eps=eps, degree=v_degree, t_e=0),
        uy_e = Expression(uy_mms, eps=eps, degree=v_degree, t_e=0),
        p_e  = Expression(p_mms, eps=eps, degree=p_degree, t_e=0),
        t_e  = Constant(0.0))


# Create a mesh here
def mesh(N, **params):
    #m = UnitSquareMesh(N, N)
    m = RectangleMesh(Point(0, 0), Point(np.pi, np.pi), N, N)
    return m


def create_bcs(NS_expressions, V, Q, sys_comp, mesh, **NS_namespace):
    external = AutoSubDomain(lambda x, b: b)
    boundary = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundary.set_all(0)
    external.mark(boundary, 1)

    bcs = dict((ui, []) for ui in sys_comp)
    bc0 = DirichletBC(V, NS_expressions["ux_e"], boundary, 1)
    bc1 = DirichletBC(V, NS_expressions["uy_e"], boundary, 1)
    bcp = DirichletBC(Q, NS_expressions["p_e"], boundary, 1)
    bcs['u0'] = [bc0]
    bcs['u1'] = [bc1]
    bcs['p']  = [bcp]

    return bcs


def body_force(V, Q, mesh, ux_mms, uy_mms, wx_mms, wy_mms, p_mms, nu, eps, t_e, **NS_namespace):
    x = SpatialCoordinate(mesh)
    u_vec = as_vector([eval(ux_mms), eval(uy_mms)])
    w_vec = as_vector([eval(wx_mms), eval(wy_mms)])
    p_    = eval(p_mms)

    f = (diff(u_vec, t_e)
         + dot(u_vec - w_vec, nabla_grad(u_vec))
         + div(p_ * Identity(2))
         - nu*div((grad(u_vec))))
         #- nu*div((grad(u_vec) + grad(u_vec).T)))

    # Check if div(u) = 0 holds
    if abs(assemble(project(div(u_vec), V)*dx)) > 1e-16:
    #if not np.sum(project(div(u_vec), V).vector().get_local() != 0) < 1:
        raise ValueError("The suggested MMS solution is not divergence free. Please change" + \
                         " the equations.")

    #from IPython import embed; embed()
    # Check if u = w holds on the boundary
    if wx_mms != "0":
        assert abs(assemble(project(eval(ux_mms) - eval(wx_mms), V)*ds)) < 5e-15
        assert abs(assemble(project(eval(uy_mms) - eval(wy_mms), V)*ds)) < 5e-15

    return f


def initialize(V, Q, q_1, q_2, NS_expressions, dt, **NS_namespace):
    NS_expressions["ux_e"].t_e = -dt
    NS_expressions["uy_e"].t_e = -dt

    ux = project(NS_expressions["ux_e"], V)
    uy = project(NS_expressions["uy_e"], V)

    q_2["u0"].vector().axpy(1, ux.vector())
    q_2["u1"].vector().axpy(1, uy.vector())

    if 'IPCS' in NS_parameters['solver']:
        NS_expressions["p_e"].t_e  = 0.5*dt
    else:
        NS_expressions["p_e"].t_e  = 0
    NS_expressions["ux_e"].t_e = 0
    NS_expressions["uy_e"].t_e = 0
    NS_expressions["wx_e"].t_e = 0
    NS_expressions["wy_e"].t_e = 0

    ux = project(NS_expressions["ux_e"], V)
    uy = project(NS_expressions["uy_e"], V)
    p_ = project(NS_expressions["p_e"], Q)

    q_1["u0"].vector().axpy(1, ux.vector())
    q_1["u1"].vector().axpy(1, uy.vector())
    q_1["p"].vector().axpy(1, p_.vector())


def pre_solve_hook(V, Q, mesh, newfolder, q_, t, velocity_degree, u_components, **NS_namespace):
    """Called prior to time loop"""
    viz_sol = XDMFFile(MPI.comm_world, path.join(newfolder, "VTK","solution.xdmf"))
    viz_sol.parameters["rewrite_function_mesh"] = True
    viz_sol.parameters["flush_output"] = True
    viz_sol.parameters["functions_share_mesh"] = True
    pmms    = Function(Q, name="p_mms")
    uxmms   = Function(V, name="ux_mms")
    uymms   = Function(V, name="uy_mms")
    wxmms   = Function(V, name="wx_mms")
    wymms   = Function(V, name="wy_mms")
    p_sol    = Function(Q, name="p_sol")
    ux_sol   = Function(V, name="ux_sol")
    uy_sol   = Function(V, name="uy_sol")
    wx_sol   = Function(V, name="wx_sol")
    wy_sol   = Function(V, name="wy_sol")
    p_error  = Function(Q, name="p_error")
    ux_error = Function(V, name="ux_error")
    uy_error = Function(V, name="uy_error")

    # Set up mesh solver
    u_mesh, v_mesh = TrialFunction(V), TestFunction(V)
    f_mesh = Function(V)

    alfa = Constant(1)
    F_mesh = (alfa * inner(grad(u_mesh), grad(v_mesh))*dx
              + inner(f_mesh, v_mesh)*dx)
    a_mesh = lhs(F_mesh)
    l_mesh = rhs(F_mesh)

    A_mesh = assemble(a_mesh)
    L_mesh = assemble(l_mesh)

    external = AutoSubDomain(lambda x, b: b)
    boundary = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundary.set_all(0)
    external.mark(boundary, 1)

    bc_x = DirichletBC(V, NS_expressions["wx_e"], boundary, 1)
    bc_y = DirichletBC(V, NS_expressions["wy_e"], boundary, 1)

    bc_mesh = dict((ui, []) for ui in u_components)
    bc_mesh["u0"] = [bc_x]
    bc_mesh["u1"] = [bc_y]

    mesh_prec = PETScPreconditioner("ilu")    # In tests "sor" is faster.
    mesh_sol = PETScKrylovSolver("gmres", mesh_prec)

    krylov_solvers = dict(monitor_convergence=False,
                          report=True,
                          error_on_nonconvergence=False,
                          nonzero_initial_guess=True,
                          maximum_iterations=20,
                          relative_tolerance=1e-8,
                          absolute_tolerance=1e-8)

    mesh_sol.parameters.update(krylov_solvers)
    coordinates = mesh.coordinates()

    if velocity_degree == 2:
        Vv = FunctionSpace(mesh, "CG", 1)
        dof_map = vertex_to_dof_map(Vv)
    else:
        Vv = None
        dof_map = vertex_to_dof_map(V)

    return dict(mesh_sol=mesh_sol, dof_map=dof_map,
                wxmms=wxmms, wymms=wymms, Vv=Vv,
                wx_sol=wx_sol, wy_sol=wy_sol,
                F_mesh=F_mesh, bc_mesh=bc_mesh, a_mesh=a_mesh, l_mesh=l_mesh,
                A_mesh=A_mesh, L_mesh=L_mesh, coordinates=coordinates, viz_sol=viz_sol,
                p_sol=p_sol, ux_sol=ux_sol, uy_sol=uy_sol, p_error=p_error,
                ux_error=ux_error, uy_error=uy_error, pmms=pmms, uxmms=uxmms, uymms=uymms)


def start_timestep_hook(t, dt, NS_expressions, **NS_namespace):
    NS_expressions["ux_e"].t_e = t
    NS_expressions["uy_e"].t_e = t
    NS_expressions["wx_e"].t_e = t
    NS_expressions["wy_e"].t_e = t

    if 'IPCS' in NS_parameters['solver']:
        NS_expressions["p_e"].t_e = t - 0.5*dt
        NS_expressions["t_e"].assign(t-dt/2)
    else:
        NS_expressions["p_e"].t_e  = t
        NS_expressions["t_e"].assign(t)


def update_prescribed_motion(t, dt, d_, d_1, w_, u_components, tstep, mesh_sol, F_mesh,
                             bc_mesh, NS_expressions, dof_map, V, Vv, A_cache,
                             a_mesh, l_mesh, A_mesh, L_mesh, mesh, coordinates, t,
                             **NS_namespace):
    P = VectorFunctionSpace(mesh, "CG", NS_namespace["velocity_degree"])
    a = project(NS_namespace["wu_"] - NS_namespace["U_AB"], P)
    print(np.abs(a.vector().get_local()).min())

    #for ui in u_components:
        # Update deformation
        #d_1[ui].vector().zero()
        #d_1[ui].vector().axpy(1, d_[ui].vector())

        # Solve for d and w
        #assemble(a_mesh, tensor=A_mesh)
        #assemble(l_mesh, tensor=L_mesh)

        #for bc in bc_mesh[ui]:
        #    bc.apply(A_mesh, L_mesh)

        #mesh_sol.solve(A_mesh, d_[ui].vector(), L_mesh)

        # Compute deformation increment
    wx = interpolate(NS_expressions["wx_e"], V)
    wy = interpolate(NS_expressions["wy_e"], V)

    w_["u0"].vector().zero()
    w_["u1"].vector().zero()
    w_["u0"].vector().axpy(1, wx.vector())
    w_["u1"].vector().axpy(1, wy.vector())

    # Move mesh
    if Vv is None:
        coordinates[:, 0] += (wx.vector().get_local()*dt)[dof_map]
        coordinates[:, 1] += (wy.vector().get_local()*dt)[dof_map]
    else:
        wx_tmp = interpolate(NS_expressions["wx_e"], Vv)
        wy_tmp = interpolate(NS_expressions["wy_e"], Vv)
        coordinates[:, 0] += (wx_tmp.vector().get_local()*dt)[dof_map]
        coordinates[:, 1] += (wy_tmp.vector().get_local()*dt)[dof_map]

    # Do we need this line?
    mesh.bounding_box_tree().build(mesh)
    A_cache.update_t(t)


def temporal_hook(t, dt, q_, viz_sol, p_e, p_error, pmms, p_sol, ux_e, ux_error, tstep,
                  uxmms, ux_sol, wx_e, wxmms, wx_sol, uy_e, uy_error, uymms, uy_sol, wy_e,
                  wymms, wy_sol, V, Q, w_, **NS_namespace):
    if tstep % 1 == 0:
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

        wx = interpolate(wx_e, V)
        wx_sol.vector().zero()
        wx_sol.vector().axpy(1, w_["u0"].vector())
        wxmms.vector().zero()
        wxmms.vector().axpy(1, wx.vector())

        uy = interpolate(uy_e, V)
        uy_sol.vector().zero()
        uy_sol.vector().axpy(1, q_["u1"].vector())
        uymms.vector().zero()
        uymms.vector().axpy(1, uy.vector())
        uy_error.vector().zero()
        uy_error.vector().axpy(1, uy.vector())
        uy_error.vector().axpy(-1, q_["u1"].vector())

        wy = interpolate(wy_e, V)
        wy_sol.vector().zero()
        wy_sol.vector().axpy(1, w_["u1"].vector())
        wymms.vector().zero()
        wymms.vector().axpy(1, wy.vector())

        viz_sol.write(ux_sol, t)
        viz_sol.write(uxmms, t)
        viz_sol.write(ux_error, t)
        viz_sol.write(wx_sol, t)
        viz_sol.write(wxmms, t)
        viz_sol.write(uy_sol, t)
        viz_sol.write(uymms, t)
        viz_sol.write(uy_error, t)
        viz_sol.write(wy_sol, t)
        viz_sol.write(wymms, t)
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
    print("L2 norm (ux-uxmms) {0:.6e}".format(errornorm(ux, q_["u0"], norm_type="l2",
                                                        degree_rise=5)))
    print("L2 norm (uy-uymms) {0:.6e}".format(errornorm(uy, q_["u1"], norm_type="l2",
                                                        degree_rise=5)))
    print("L2 norm (p-pmms)   {0:.6e}".format(errornorm(p, q_["p"], norm_type="l2",
                                                        degree_rise=5)))
    print(" ")
