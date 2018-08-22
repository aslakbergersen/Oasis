from ..NSfracStep import *
import sys
import pickle
from os import makedirs
import random
import numpy as np
import meshio
import pygmsh

set_log_level(99)
parameters["allow_extrapolation"] = True


def get_d_and_h(N):
    area = np.loadtxt("a.txt", delimiter=", ")
    ratio = np.loadtxt("ratio.txt", delimiter=", ")

    # 10 000 timesteps per cycle
    t = np.linspace(0, 1.06, N)

    area_new = np.zeros(t.shape[0])
    ratio_new = np.zeros(t.shape[0])

    for i in range(t.shape[0]):
        if t[i] == 1.06 or t[i] < max(area[:,0].min(), ratio[:,0].min()):
            area_new[i] = np.pi * 3.4 * 6.8 / 4
            ratio_new[i] = 2
            continue
        else:
            #from IPython import embed; embed()
            area_x_larger = area[area[:,0] > t[i], 0].min()
            area_x_smaller = area[area[:,0] <= t[i], 0].max()
            ratio_x_larger = ratio[ratio[:,0] > t[i], 0].min()
            ratio_x_smaller = ratio[ratio[:,0] <= t[i], 0].max()

            area_index_smaller = np.where(area[:,0] == area_x_smaller)[0][0]
            area_index_larger = np.where(area[:,0] == area_x_larger)[0][0]
            ratio_index_smaller = np.where(ratio[:,0] == ratio_x_smaller)[0][0]
            ratio_index_larger = np.where(ratio[:,0] == ratio_x_larger)[0][0]

            area_x_dist = t[i] - area_x_smaller
            ratio_x_dist = t[i] - ratio_x_smaller

            a_area = ((area[area_index_larger, 1] - area[area_index_smaller, 1]) /
                        (area_x_larger - area_x_smaller))
            a_ratio = ((ratio[ratio_index_larger, 1] - ratio[ratio_index_smaller, 1]) /
                        (ratio_x_larger - ratio_x_smaller))

            area_extrapolate = area_x_dist * a_area
            ratio_extrapolate = ratio_x_dist * a_ratio

            area_new[i] = area[area_index_smaller, 1] + area_extrapolate
            ratio_new[i] = ratio[ratio_index_smaller, 1] + ratio_extrapolate

    ratio_new[ratio_new > 2] = 2
    area_new[area_new < np.pi * 3.4 * 6.8 / 4] = np.pi * 3.4 * 6.8 / 4

    H = np.sqrt(4 * area_new * ratio_new / np.pi)
    H = H / H[0]

    D = H / ratio_new
    D = D / D[0]

    return D, H


def problem_parameters(commandline_kwargs, NS_parameters, NS_expressions, **NS_namespace):
    if "restart_folder" in commandline_kwargs.keys():
        restart_folder = commandline_kwargs["restart_folder"]
        restart_folder = path.join(getcwd(), restart_folder)

        f = open(path.join(path.dirname(path.abspath(__file__)), restart_folder,
                           'params.dat'), 'r')
        NS_parameters.update(cPickle.load(f))
        NS_parameters['T'] = 10
        NS_parameters['restart_folder'] = restart_folder
        globals().update(NS_parameters)

    else:
        T = 2.12
        N = 1000
        dt = 1.06 / N
        nu = 0.035
        NS_parameters.update(
            checkpoint = 1000,
            save_step = 10,
            check_flux = 2,
            print_intermediate_info = 1000,
            nu = nu,
            T = T,
            N = N,
            dt = dt,
            velocity_degree = 1,
            folder = "heart_results",
            lc=0.05,
            mesh_path = "/Users/Aslak/Dropbox/Work/FEniCS/prescribed/Oasis/oasis/mesh/heart.xdmf",
            mesh_function_path = "/Users/Aslak/Dropbox/Work/FEniCS/prescribed/Oasis/oasis/mesh/heart_cell_markers.xdmf",
            use_krylov_solvers = True,
            )


def mesh(mesh_path, mesh_function_path, lc, **NS_namespace):
    # Inspired from
    # https://gist.github.com/michalhabera/bbe8a17f788192e53fd758a67cbf3bed
    geom = pygmsh.built_in.geometry.Geometry()

    r1 = 1.7*2
    r2 = 3.8*2

    # Create geometry
    p0 = geom.add_point([ 0, 0, 0], lcar=lc)
    p1 = geom.add_point([ r1, 0, 0], lcar=lc)
    p2 = geom.add_point([ 0, -2*r2, 0], lcar=lc)
    p3 = geom.add_point([-r1, 0, 0], lcar=lc)

    p4 = geom.add_point([ 1.615, 0, 0], lcar=lc)
    p5 = geom.add_point([-0.085, 0, 0], lcar=lc)
    p6 = geom.add_point([-0.272, 0, 0], lcar=lc)
    p7 = geom.add_point([-1.632, 0, 0], lcar=lc)

    l0 = geom.add_ellipse_arc(p1, p0, p2, p2)
    l1 = geom.add_ellipse_arc(p2, p0, p3, p3)
    #l2 = geom.add_line(p3, p1)
    l3 = geom.add_line(p3, p7)
    l4 = geom.add_line(p7, p6)
    l5 = geom.add_line(p6, p5)
    l6 = geom.add_line(p5, p4)
    l7 = geom.add_line(p4, p1)

    #ll = geom.add_line_loop(lines=[l0, l1, l2])
    ll = geom.add_line_loop(lines=[l7, l0, l1, l3, l4, l5, l6])
    ps = geom.add_plane_surface(ll)

    # Tag line and surface
    geom.add_physical_line(lines=l4, label=1)
    geom.add_physical_line(lines=l6, label=2)
    geom.add_physical_line(lines=[l7, l0, l1, l3, l5], label=3)
    geom.add_physical_surface(surfaces=ps, label=4)

    # Mesh surface
    points, cells, point_data, cell_data, field_data = pygmsh.generate_mesh(geom)

    # Write, then read mesh and MeshFunction
    meshio.write(mesh_path, meshio.Mesh(
                 points=points,
                 cells={"triangle": cells["triangle"]}))

    meshio.write(mesh_function_path, meshio.Mesh(
                 points=points,
                 cells={"line": cells["line"]},
                 cell_data={"line": {"boundary": cell_data["line"]["gmsh:physical"]}}
                ))

    mesh = Mesh()
    with XDMFFile(MPI.comm_world, mesh_path) as infile:
        infile.read(mesh)

    return mesh


class Wall(UserExpression):
    def __init__(self, tstep, comp, DH, dt, N, **kwargs):
        self.tstep = tstep
        self.comp = comp
        self.DH = DH
        self.dt = dt
        self.N = N
        self.ds = ds
        super().__init__(**kwargs)

    def eval(self, values, x):
        tstep = self.tstep % self.N
        values[:] = (self.DH[tstep] - self.DH[tstep-1]) / self.dt * x[self.comp]


class Wall2(UserExpression):
    def __init__(self, tstep, D, H, dt, N, **kwargs):
        self.tstep = tstep
        self.D = D
        self.H = H
        self.dt = dt
        self.N = N
        super().__init__(**kwargs)

    def eval(self, values, x):
        tstep = self.tstep % self.N
        values[:] = [(self.D[tstep] - self.D[tstep-1]) / self.dt * x[0],
                     (self.H[tstep] - self.H[tstep-1]) / self.dt * x[1], 0]


class Inlet(UserExpression):
    def __init__(self, tstep, D, H, dt, ds, N, **kwargs):
        self.tstep = tstep % N
        self.D = D
        self.H = H
        self.dt = dt
        self.ds = ds
        self.N = N

        self.u = 0

        super().__init__(**kwargs)

    def update(self, tstep):
        tstep = tstep % self.N
        Q = (np.pi / 4 * self.D[tstep] * self.H[tstep]
                     * ((self.D[tstep] - self.D[tstep-1]) / self.dt / self.D[tstep]
                       + (self.H[tstep] - self.H[tstep - 1]) / self.dt / self.H[tstep]))
        Q = 0 if 0 >= Q else Q
        length = assemble(Constant(1) * self.ds(2))

        print("inlet", Q/length)
        self.u = Q / length

    def eval(self, values, x):
        values[:] = self.u


class Outlet(UserExpression):
    def __init__(self, tstep, D, H, dt, ds,N, **kwargs):
        self.tstep = tstep % N
        self.D = D
        self.H = H
        self.dt = dt
        self.ds = ds
        self.N = N

        self.u = 0

        super().__init__(**kwargs)

    def update(self, tstep):
        tstep = tstep % self.N
        Q = (np.pi / 4 * self.D[tstep] * self.H[tstep]
                     * ((self.D[tstep] - self.D[tstep-1]) / self.dt / self.D[tstep]
                       + (self.H[tstep] - self.H[tstep - 1]) / self.dt / self.H[tstep]))
        Q = 0 if 0 <= Q else Q
        length = assemble(Constant(1) * self.ds(2))
        print("outlet", Q/length)
        self.u = Q / length

    def eval(self, values, x):
        values[:] = -self.u


def create_bcs(V, Q, w_, sys_comp, mesh_function_path, u_components, mesh, newfolder,
               NS_expressions, tstep, dt, N, **NS_namespace):
    info_red("Creating boundary conditions")

    inlet = AutoSubDomain(lambda x, b: b and -0.085 - DOLFIN_EPS <= x[0] <= 1.615 +
                          DOLFIN_EPS and x[1] > 0 - DOLFIN_EPS)
    outlet = AutoSubDomain(lambda x, b: b and -1.632 - DOLFIN_EPS <= x[0] <= -0.272 +
                           DOLFIN_EPS and x[1] > 0 - DOLFIN_EPS)
    walls = AutoSubDomain(lambda x, b: b and not ((-1.632 + DOLFIN_EPS <= x[0] <= -0.272 -
                                                   DOLFIN_EPS and x[1] > 0 - DOLFIN_EPS)
                                                  or (-0.085 + DOLFIN_EPS <= x[0] <=
                                                      -1.615 - DOLFIN_EPS
                                                      and x[1] < 0 - DOLFIN_EPS)))
    boundary = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundary.set_all(0)
    walls.mark(boundary, 1)
    outlet.mark(boundary, 2)
    inlet.mark(boundary, 3)

    D, H = get_d_and_h(N)
    ds = Measure('ds', domain=mesh, subdomain_data=boundary)
    wall0 = Wall(tstep, 0, D, dt, N, element=V.ufl_element())
    wall1 = Wall(tstep, 1, H, dt, N, element=V.ufl_element())
    inlet = Inlet(tstep, D, H, dt, ds, N, element=V.ufl_element())
    outlet = Outlet(tstep, D, H, dt, ds, N, element=V.ufl_element())

    NS_expressions["wall0"] = wall0
    NS_expressions["wall1"] = wall1
    NS_expressions["inlet"] = inlet
    NS_expressions["outlet"] = outlet

    bcs = dict((ui, []) for ui in sys_comp)
    bcu_wall_x = DirichletBC(V, wall0, boundary, 1)
    bcu_wall_y = DirichletBC(V, wall1, boundary, 1)

    bcu_in_x = DirichletBC(V, Constant(0), boundary, 3)
    #bcu_in_y = DirichletBC(V, inlet, boundary, 3)
    bcu_in_y = DirichletBC(V, inlet, boundary, 3)

    bcu_out_x = DirichletBC(V, Constant(0), boundary, 2)
    #bcu_out_y = DirichletBC(V, outlet, boundary, 2)

    bcp_in = DirichletBC(Q, Constant(0), boundary, 2)
    bcp_out = DirichletBC(Q, Constant(0), boundary, 3)

    bcs['u0'] = [bcu_in_x, bcu_out_x, bcu_wall_x]
    bcs['u1'] = [bcu_in_y, bcu_wall_y] # bcu_out_y
    bcs["p"] = [bcp_in]

    return bcs


def pre_solve_hook(W, V, u_, mesh, newfolder, T, d_, velocity_degree, tstep, dt, N,
                   **NS_namespace):
    """Called prior to time loop"""
    viz_d = XDMFFile(MPI.comm_world, path.join(newfolder, "VTK", "deformation.xdmf"))
    viz_u = XDMFFile(MPI.comm_world, path.join(newfolder, "VTK", "velocity.xdmf"))
    viz_p = XDMFFile(MPI.comm_world, path.join(newfolder, "VTK", "pressure.xdmf"))
    viz_w = XDMFFile(MPI.comm_world, path.join(newfolder, "VTK", "mesh_velocity.xdmf"))

    for viz in [viz_d, viz_u, viz_p, viz_w]:
        viz.parameters["rewrite_function_mesh"] = True
        viz.parameters["flush_output"] = True

    inlet = AutoSubDomain(lambda x, b: b and -0.085 - DOLFIN_EPS <= x[0] <= 1.615 +
                          DOLFIN_EPS and x[1] > 0 - DOLFIN_EPS)
    outlet = AutoSubDomain(lambda x, b: b and -1.632 - DOLFIN_EPS <= x[0] <= -0.272 +
                           DOLFIN_EPS and x[1] > 0 - DOLFIN_EPS)
    walls = AutoSubDomain(lambda x, b: b)
    #and not ((-1.632 + DOLFIN_EPS <= x[0] <= -0.272 -
    #                                               DOLFIN_EPS and x[1] > 0 - DOLFIN_EPS)
    #                                              or (-0.085 + DOLFIN_EPS <= x[0] <=
    #                                                  -1.615 - DOLFIN_EPS
    #                                                  and x[1] < 0 - DOLFIN_EPS)))
    boundary = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundary.set_all(0)
    walls.mark(boundary, 1)
    outlet.mark(boundary, 2)
    inlet.mark(boundary, 3)

    f = File("test.pvd")
    f << boundary

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

    D, H = get_d_and_h(N)
    wall = Wall2(tstep, D, H, dt, N, element=W.ufl_element())
    NS_expressions["wall"] = wall


    walls_bc = DirichletBC(W, wall, boundary, 1)
    inlet_bc = DirichletBC(W, Constant((0, 0, 0)), boundary, 2)
    outlet_bc = DirichletBC(W, Constant((0, 0, 0)), boundary, 3)
    bc_mesh = [inlet_bc, outlet_bc, walls_bc]

    mesh_prec = PETScPreconditioner("ilu")  # in tests sor are faster .. 
    mesh_sol = PETScKrylovSolver("gmres", mesh_prec)
    w_vec = Function(W)

    krylov_solvers = dict(monitor_convergence=False,
                          report=False,
                          error_on_nonconvergence=False,
                          nonzero_initial_guess=True,
                          maximum_iterations=200,
                          relative_tolerance=1e-10,
                          absolute_tolerance=1e-10)

    mesh_sol.parameters.update(krylov_solvers)

    tstep = np.argmax(H)

    return dict(viz_p=viz_p, viz_u=viz_u, viz_d=viz_d, u_vec=u_vec, mesh_sol=mesh_sol,
                F_mesh=F_mesh, bc_mesh=bc_mesh, w_vec=w_vec, viz_w=viz_w,
                a_mesh=a_mesh, l_mesh=l_mesh, A_mesh=A_mesh, L_mesh=L_mesh, tstep=tstep)


def update_prescribed_motion(t, dt, d_, d_1, w_, u_components, tstep, mesh_sol, F_mesh,
                             bc_mesh, w_vec, NS_expressions,
                             a_mesh, l_mesh, A_mesh, L_mesh,
                             **NS_namespace):
    # Update time
    for key, value in NS_expressions.items():
        if "let" in key:
            value.update(tstep)
        else:
            value.tstep = tstep
    #NS_expressions["wall1"].tstep = tstep
    #NS_expressions["wall"].tstep = tstep
    #NS_expressions["inlet"].tstep = tstep
    #NS_expressions["outlet"].tstep = tstep

    # Read deformation
    d_1.vector().zero()
    #d_1.vector().axpy(1, d_.vector())

    # Solve for d and w
    assemble(a_mesh, tensor=A_mesh)
    assemble(l_mesh, tensor=L_mesh)

    for bc in bc_mesh:
        bc.apply(A_mesh, L_mesh)

    mesh_sol.solve(A_mesh, d_.vector(), L_mesh)

    w_vec.vector().zero()
    w_vec.vector().axpy(1, d_.vector())
    #w_vec.vector().axpy(-1, d_1.vector())

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
