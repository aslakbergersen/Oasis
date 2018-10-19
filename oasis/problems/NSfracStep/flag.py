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
        T = 50
        dt = 0.025
        nu = 0.01
        NS_parameters.update(
            checkpoint = 1000,
            save_step = 10e10,
            print_intermediate_info = 1000,
            # Geometrical parameters
            H = 12,
            L = 23,
            b_dist = 4.5,
            b_h = 1,
            b_l = 1,
            f_l = 4.5,
            f_h = 0.06,
            nu = nu,
            T = T,
            dt = dt,
            velocity_degree = 1,
            folder = "flag_results",
            lc=0.05,
            mesh_path = "/Users/Aslak/Dropbox/Work/FEniCS/prescribed/Oasis/oasis/mesh/flag.xdmf",
            use_krylov_solvers = True,
            )


def mesh(mesh_path, lc, H, L, b_dist, b_h, b_l, f_l, f_h, **NS_namespace):
    # Initialize geometry
    geom = pygmsh.built_in.geometry.Geometry()

    # Surounding box
    b0 = geom.add_point([0, 0, 0], lcar=lc*2)
    b1 = geom.add_point([L, 0, 0], lcar=lc*2)
    b2 = geom.add_point([L, H, 0], lcar=lc*2)
    b3 = geom.add_point([0, H, 0], lcar=lc*2)

    # Inner geometry
    f0 = geom.add_point([b_dist, H / 2 - b_h / 2, 0], lcar=lc/3)
    f1 = geom.add_point([b_dist + b_l, H / 2 - b_h / 2, 0], lcar=lc/3)
    f2 = geom.add_point([b_dist + b_l, H / 2 - f_h / 2, 0], lcar=lc/3)
    f3 = geom.add_point([b_dist + b_l + f_l, H / 2 - f_h / 2, 0], lcar=lc/3)
    f4 = geom.add_point([b_dist + b_l + f_l, H / 2 + f_h / 2, 0], lcar=lc/3)
    f5 = geom.add_point([b_dist + b_l, H / 2 + f_h / 2, 0], lcar=lc/3)
    f6 = geom.add_point([b_dist + b_l, H / 2 + b_h / 2, 0], lcar=lc/3)
    f7 = geom.add_point([b_dist, H / 2 + b_h / 2, 0], lcar=lc/3)

    # Surounding box
    l0 = geom.add_line(b0, b1)
    l1 = geom.add_line(b1, b2)
    l2 = geom.add_line(b2, b3)
    l3 = geom.add_line(b3, b0)

    l4 = geom.add_line(f0, f1)
    l5 = geom.add_line(f1, f2)
    l6 = geom.add_line(f2, f3)
    l7 = geom.add_line(f3, f4)
    l8 = geom.add_line(f4, f5)
    l9 = geom.add_line(f5, f6)
    l10 = geom.add_line(f6, f7)
    l11 = geom.add_line(f7, f0)

    ll_outer = geom.add_line_loop(lines=[l0, l1, l2, l3])
    ll_inner = geom.add_line_loop(lines=[l4, l5, l6, l7, l8, l9, l10, l11])
    ps = geom.add_plane_surface(ll_outer, [ll_inner])

    # Mesh surface
    points, cells, point_data, cell_data, field_data = pygmsh.generate_mesh(geom)

    # Write, then read mesh and MeshFunction
    meshio.write(mesh_path, meshio.Mesh(
                 points=points,
                 cells={"triangle": cells["triangle"]}))

    mesh = Mesh()
    with XDMFFile(MPI.comm_world, mesh_path) as infile:
        infile.read(mesh)

    return mesh


class Flag_y(UserExpression):
    def __init__(self, t, T_end, factor, wall_x, **kwargs):
        self.factor = factor
        self.T_end = T_end
        self.t = t
        self.A = 0
        self.wall_x = wall_x

        super().__init__(**kwargs)

    def get_A(self):
        #if self.t < 2:
        #    return  0
        #else:
        # FIXME: put two back
        return self.factor * np.sin(2 * np.pi * (self.t - 2 + 2) / self.T_end)

    def update(self, t):
        self.t = t
        self.A = self.get_A()

    def external_eval(self, x):
        values = [0]
        self.eval(values, x)
        return values

    def eval(self, values, x):
        # FIXME: put self.wall_x back
        values[:] = [(x[0] - 5.5)**2 * self.A]


class Flag_x(UserExpression):
    def __init__(self, t, T_end, factor, **kwargs):
        self.factor = factor # A * C
        self.T_end = T_end
        self.t = t
        self.A = 0
        super().__init__(**kwargs)

    def get_A(self):
        #if self.t < 2:
        #    return 0
        #else:
        # FIXME: set two back
        return - self.factor * np.sin(2 * np.pi * (self.t - 2 + 2) / self.T_end)

    def update(self, t):
        self.t = t
        self.A = self.get_A()

    def angle(self, x):
        return np.arccos((x[0] - 5.5) / ((x[0] - 5.5)**2 + x[1]**2))

    def external_eval(self, x):
        values = [0]
        self.eval(values, x)
        return values

    def eval(self, values, x):
        tetha = self.angle(x)
        values[:] = [self.A * (x[0] - 5.5)**2 * np.tan(tetha) - (x[1] - 6) * np.sin(tetha)] # y not centered?
        #print(self.A * (x[0] - 5.5)**2 * np.tan(tetha) - (x[1] - 6) * np.sin(tetha))


def pre_boundary_condition(H, b_dist, b_l, f_h, b_h, L, mesh, **NS_namespace):
    inlet = AutoSubDomain(lambda x, b: b and x[0] <= DOLFIN_EPS)
    walls = AutoSubDomain(lambda x, b: b and (near(x[1], 0) or near(x[1], H)))
    box = AutoSubDomain(lambda x, b: b and (1 <= x[1] <= 3*H/4) and (1 <= x[0] <= b_dist +
                                                                     b_l + DOLFIN_EPS * 1000))
    flag = AutoSubDomain(lambda x, b: b and (H / 2 - f_h - DOLFIN_EPS <= x[1] <= H /
                                             2 + f_h + DOLFIN_EPS) and (b_dist + b_l -
                                                                      DOLFIN_EPS*1000 <= x[0] <= L - 1))
    outlet = AutoSubDomain(lambda x, b: b and (x[0] > L - DOLFIN_EPS * 1000))

    boundary = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundary.set_all(0)

    inlet.mark(boundary, 1)
    walls.mark(boundary, 2)
    box.mark(boundary, 3)
    flag.mark(boundary, 4)
    outlet.mark(boundary, 5)

    return dict(boundary=boundary)


def create_bcs(V, Q, w_, sys_comp, u_components, mesh, newfolder,
               boundary, NS_expressions, tstep, dt, **NS_namespace):
    info_red("Creating boundary conditions")

    flag_x = Flag_x(0, 10, 0.075, element=V.ufl_element())
    flag_y = Flag_y(0, 10, 0.01875, flag_x, element=V.ufl_element())

    NS_expressions["flag_x"] = flag_x
    NS_expressions["flag_y"] = flag_y

    bcu_in_x = DirichletBC(V, Constant(1), boundary, 1)
    bcu_in_y = DirichletBC(V, Constant(0), boundary, 1)

    bcu_wall = DirichletBC(V, Constant(0), boundary, 2)
    bcu_box = DirichletBC(V, Constant(0), boundary, 3)

    bcu_flag_x = DirichletBC(V, Constant(0), boundary, 4) #flag_x, boundary, 4)
    bcu_flag_y = DirichletBC(V, flag_y, boundary, 4)

    bcp_out = DirichletBC(Q, Constant(0), boundary, 5)

    bcs = dict((ui, []) for ui in sys_comp)
    bcs['u0'] = [bcu_flag_x, bcu_in_x, bcu_wall, bcu_box]
    bcs['u1'] = [bcu_flag_y, bcu_in_y, bcu_wall, bcu_box] # bcu_out_y
    bcs["p"] = [bcp_out]

    return bcs


def pre_solve_hook(V, u_, mesh, newfolder, T, d_, velocity_degree, tstep, dt, L,
                   b_dist, b_l, H, f_h, u_components, boundary, **NS_namespace):
    """Called prior to time loop"""
    viz_d = XDMFFile(MPI.comm_world, path.join(newfolder, "VTK", "deformation.xdmf"))
    viz_u = XDMFFile(MPI.comm_world, path.join(newfolder, "VTK", "velocity.xdmf"))
    viz_p = XDMFFile(MPI.comm_world, path.join(newfolder, "VTK", "pressure.xdmf"))
    viz_w = XDMFFile(MPI.comm_world, path.join(newfolder, "VTK", "mesh_velocity.xdmf"))

    for viz in [viz_d, viz_u, viz_p, viz_w]:
        viz.parameters["rewrite_function_mesh"] = True
        viz.parameters["flush_output"] = True

    Vv = VectorFunctionSpace(mesh, "CG", velocity_degree)
    DG = FunctionSpace(mesh, "DG", velocity_degree)
    u_vec = Function(Vv, name="u")

    # Set up mesh solver
    u_mesh, v_mesh = TrialFunction(V), TestFunction(V)
    f_mesh = Function(V)

    A = CellDiameter(mesh)
    D = project(A, DG)
    D_arr = D.vector().get_local()

    # Note: Double check if alfa changes when mesh moves. I do not think so as long as
    # project is here and not inside time loop
    alfa = Constant(D_arr.max()**3 - D_arr.min()**3) / D**3
    F_mesh = (alfa * inner(grad(u_mesh), grad(v_mesh))*dx
              + inner(f_mesh, v_mesh)*dx)
    a_mesh = lhs(F_mesh)
    l_mesh = rhs(F_mesh)

    A_mesh = assemble(a_mesh)
    L_mesh = assemble(l_mesh)

    flag_bc_x = DirichletBC(V, NS_expressions["flag_x"], boundary, 4)
    flag_bc_y = DirichletBC(V, NS_expressions["flag_y"], boundary, 4)
    rigid_bc_in = DirichletBC(V, Constant(0), boundary, 1)
    rigid_bc_walls = DirichletBC(V, Constant(0), boundary, 2)
    rigid_bc_out = DirichletBC(V, Constant(0), boundary, 3)
    rigid_bc_box = DirichletBC(V, Constant(0), boundary, 5)

    bc_mesh = dict((ui, []) for ui in u_components)
    rigid_bc = [rigid_bc_in, rigid_bc_walls, rigid_bc_out, rigid_bc_box]
    bc_mesh["u0"] = [flag_bc_x] + rigid_bc
    bc_mesh["u1"] = [flag_bc_y] + rigid_bc

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
    dof_map = vertex_to_dof_map(V)

    return dict(viz_p=viz_p, viz_u=viz_u, viz_d=viz_d, u_vec=u_vec, mesh_sol=mesh_sol,
                F_mesh=F_mesh, bc_mesh=bc_mesh, viz_w=viz_w, dof_map=dof_map,
                a_mesh=a_mesh, l_mesh=l_mesh, A_mesh=A_mesh, L_mesh=L_mesh,
                coordinates=coordinates)


def update_prescribed_motion(t, dt, d_, d_1, w_, u_components, tstep, mesh_sol, F_mesh,
                             bc_mesh, NS_expressions, dof_map,
                             a_mesh, l_mesh, A_mesh, L_mesh, mesh, coordinates,
                             **NS_namespace):
    # Update time
    for key, value in NS_expressions.items():
        if key != "uv":
            value.update(t)

    for ui in u_components:
        # Update deformation
        d_1[ui].vector().zero()
        d_1[ui].vector().axpy(1, d_[ui].vector())

        # Solve for d and w
        assemble(a_mesh, tensor=A_mesh)
        assemble(l_mesh, tensor=L_mesh)

        for bc in bc_mesh[ui]:
            bc.apply(A_mesh, L_mesh)

        mesh_sol.solve(A_mesh, d_[ui].vector(), L_mesh)

        # Compute deformation increment
        w_[ui].vector().zero()
        w_[ui].vector().axpy(1/dt, d_[ui].vector())
        w_[ui].vector().axpy(-1/dt, d_1[ui].vector())

        # Move mesh
        coordinates[:, int(ui[-1])] += (w_[ui].vector().get_local()*dt)[dof_map]

    # Do we need this line?
    mesh.bounding_box_tree().build(mesh)


def temporal_hook(t, d_, w_, q_, f, tstep, viz_u, viz_d, viz_p, u_components,
                  u_vec, viz_w, **NS_namespace):
    assign(u_vec.sub(0), q_["u0"])
    assign(u_vec.sub(1), q_["u1"])

    #viz_d.write(d_, t)
    viz_u.write(u_vec, t)
    viz_p.write(q_["p"], t)
    viz_w.write(w_["u0"], t)
    viz_w.write(w_["u1"], t)

    print("Time:", round(t, 4), "u:", u_vec((12, 6, 0))) # #, file="output_flag.txt")
