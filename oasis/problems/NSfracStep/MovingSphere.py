from ..NSfracStep import *
import pickle
from mshr import Rectangle, Circle, generate_mesh

set_log_level(99)


def problem_parameters(commandline_kwargs, NS_parameters, NS_expressions, **NS_namespace):
    if "restart_folder" in commandline_kwargs.keys():
        restart_folder = commandline_kwargs["restart_folder"]
        restart_folder = path.join(getcwd(), restart_folder)

        f = open(path.join(path.dirname(path.abspath(__file__)), restart_folder,
                           'params.dat'), 'r')
        NS_parameters.update(pickle.load(f))
        NS_parameters['T'] = 10
        NS_parameters['restart_folder'] = restart_folder
        globals().update(NS_parameters)

    else:
        NS_parameters.update(
            checkpoint = 1000,
            save_step = 10e10,
            print_intermediate_info = 100,

            # Geometrical parameters
            H = 12,
            L = 26,
            #s_dist = 6,
            #s_d = 1,
            nu = 0.01,
            T = 120,
            dt = 0.01,
            velocity_degree = 1,
            pressure_degree = 1,
            mesh_path="mesh/circle.xdmf",
            folder = "moving_sphere_results",
            use_krylov_solvers = True,
            )


def mesh(mesh_path, **NS_namespace):
    m = Mesh()
    f = XDMFFile(mesh_path)
    f.read(m)

    return m


def pre_boundary_condition(H, L, mesh, **NS_namespace):
    # Mark geometry
    inlet = AutoSubDomain(lambda x, b: b and x[0] <= DOLFIN_EPS)
    walls = AutoSubDomain(lambda x, b: b and (near(x[1], 0) or near(x[1], H)))
    circle = AutoSubDomain(lambda x, b: b and (H/4 <= x[1] <= 3*H/4) and (1 <= x[0] <= L/2))
    outlet = AutoSubDomain(lambda x, b: b and (x[0] > L - DOLFIN_EPS * 1000))

    boundary = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundary.set_all(0)

    inlet.mark(boundary, 1)
    walls.mark(boundary, 2)
    circle.mark(boundary, 3)
    outlet.mark(boundary, 4)

    return dict(boundary=boundary)


def create_bcs(V, Q, w_, sys_comp, u_components, mesh, newfolder,
               boundary, NS_expressions, tstep, dt, **NS_namespace):
    info_red("Creating boundary conditions")

    NS_expressions["circle_x"] = Constant(0)
    NS_expressions["circle_y"] = Constant(0)

    bcu_in_x = DirichletBC(V, Constant(1), boundary, 1)
    bcu_in_y = DirichletBC(V, Constant(0), boundary, 1)

    bcu_wall = DirichletBC(V, Constant(0), boundary, 2)
    bcu_circle = DirichletBC(V, Constant(0), boundary, 3)

    bcp_out = DirichletBC(Q, Constant(0), boundary, 4)

    bcs = dict((ui, []) for ui in sys_comp)
    bcs['u0'] = [bcu_circle, bcu_in_x, bcu_wall]
    bcs['u1'] = [bcu_circle, bcu_in_y, bcu_wall] # bcu_out_y
    bcs["p"] = [bcp_out]

    return bcs


def pre_solve_hook(V, p_, u_, nu, mesh, newfolder, velocity_degree, assemble_matrix, x_,
                   u_components, boundary, **NS_namespace):
    """Called prior to time loop"""
    viz_d = XDMFFile(MPI.comm_world, path.join(newfolder, "VTK", "deformation.xdmf"))
    viz_u = XDMFFile(MPI.comm_world, path.join(newfolder, "VTK", "velocity.xdmf"))
    viz_p = XDMFFile(MPI.comm_world, path.join(newfolder, "VTK", "pressure.xdmf"))
    viz_w = XDMFFile(MPI.comm_world, path.join(newfolder, "VTK", "mesh_velocity.xdmf"))

    for viz in [viz_d, viz_u, viz_p, viz_w]:
        viz.parameters["rewrite_function_mesh"] = True
        viz.parameters["flush_output"] = True
        viz.parameters["functions_share_mesh"] = True

    # Facet normals
    n = FacetNormal(mesh)

    # Set up mesh solver
    u_mesh, v_mesh = TrialFunction(V), TestFunction(V)

    DG = FunctionSpace(mesh, "DG", velocity_degree)
    A = CellDiameter(mesh)
    D = project(A, DG)
    D_arr = D.vector().get_local()

    # Note: Set alfa to something 1/distace, computed by laplace.
    alfa = Constant(D_arr.max()**3 - D_arr.min()**3) / D**3
    a_mesh = inner(alfa*grad(u_mesh), grad(v_mesh))*dx

    L_mesh = dict((ui, assemble(Function(V)*v_mesh*dx)) for ui in u_components)

    # Inlet, walls
    rigid_bc_in = DirichletBC(V, Constant(0), boundary, 1)
    rigid_bc_walls = DirichletBC(V, Constant(0), boundary, 2)
    circle_bc_x = DirichletBC(V, NS_expressions["circle_x"], boundary, 3)
    circle_bc_y = DirichletBC(V, NS_expressions["circle_y"], boundary, 3)
    rigid_bc_out = DirichletBC(V, Constant(0), boundary, 4)

    bc_mesh = dict((ui, []) for ui in u_components)
    rigid_bc = [rigid_bc_in, rigid_bc_walls, rigid_bc_out]
    bc_mesh["u0"] = [circle_bc_x] + rigid_bc
    bc_mesh["u1"] = [circle_bc_y] + rigid_bc

    mesh_prec = PETScPreconditioner("hypre_amg")    # In tests "sor" is faster. ilu
    mesh_sol = PETScKrylovSolver("gmres", mesh_prec)

    ds = Measure("ds", subdomain_data=boundary)

    R = VectorFunctionSpace(mesh, 'R', 0)
    c = TestFunction(R)
    tau = -p_ * Identity(2) + nu * (grad(u_) + grad(u_).T)
    forces = dot(dot(tau, n), c) * ds(3)

    krylov_solvers = dict(monitor_convergence=False,
                          report=False,
                          error_on_nonconvergence=False,
                          nonzero_initial_guess=True,
                          maximum_iterations=20,
                          relative_tolerance=1e-8,
                          absolute_tolerance=1e-8)

    mesh_sol.parameters.update(krylov_solvers)
    coordinates = mesh.coordinates()
    dof_map = vertex_to_dof_map(V)

    return dict(viz_p=viz_p, viz_u=viz_u, viz_w=viz_w,
                mesh_sol=mesh_sol, bc_mesh=bc_mesh,
                dof_map=dof_map, a_mesh=a_mesh, #A_mesh=A_mesh,
                L_mesh=L_mesh, n=n, forces=forces,
                coordinates=coordinates)


def update_prescribed_motion(t, dt, wx_, w_, u_components, tstep, mesh_sol,
                             bc_mesh, NS_expressions, dof_map, A_cache,
                             a_mesh, L_mesh, mesh, coordinates, **NS_namespace):
    move = False
    for ui in u_components:
        # Solve for d and w
        A_mesh = A_cache[(a_mesh, tuple(bc_mesh[ui]))]
        [bc.apply(L_mesh[ui]) for bc in bc_mesh[ui]]
        mesh_sol.solve(A_mesh, wx_[ui], L_mesh[ui])

        # Move mesh
        arr = w_[ui].vector().get_local()
        if 1e-15 < abs(arr.min()) + abs(arr.max()):
            coordinates[:, int(ui[-1])] += (arr*dt)[dof_map]
            move = True

    if move:
        mesh.bounding_box_tree().build(mesh)
        A_cache.update_t(t)

    return move


def temporal_hook(t, tstep, w_, q_, forces, f, viz_u, viz_p, viz_w, **NS_namespace):
    if tstep % 10 == 0:
        viz_u.write(q_["u0"], t)
        viz_u.write(q_["u1"], t)
        viz_p.write(q_["p"], t)
        viz_w.write(w_["u0"], t)
        viz_w.write(w_["u1"], t)

    Dr = assemble(forces).get_local() * 0.025
    print("Time:", round(t, 4), "Drag", Dr[0], "Lift", Dr[1])
