from ..NSfracStep import *
from math import pi
from fenicstools import StatisticsProbes
from numpy import array, linspace
import sys

# Override some problem specific parameters
recursive_update(NS_parameters,
                 dict(mu=0.0035,
                      nu=0.0035 / 1056.,
                      T=1000,
                      dt=0.001,
                      folder="nozzle_results",
                      case=500,
                      save_tstep=1000,
                      checkpoint=1000,
                      check_steady=100,
                      velocity_degree=1,
                      mesh_path="mesh/mesh_medium.xml",  #"mesh/8M_nozzle.xml",
                      print_intermediate_info=1000,
                      use_lumping_of_mass_matrix=True,
                      low_memory_version=True,
                      use_krylov_solvers=True,
                      krylov_solvers=dict(monitor_convergence=False,
                                          relative_tolerance=1e-8)))


def mesh(mesh_path, **NS_namespace):
    return Mesh(mesh_path)


# walls = 0
def walls(x, on_boundary):
    return on_boundary \
        and x[2] < 0.320 - DOLFIN_EPS \
        and x[2] > -0.18269 + DOLFIN_EPS


# inlet = 1
def inlet(x, on_boundary):
    return on_boundary and x[2] < -0.18269 + DOLFIN_EPS


# outlet = 2
def outlet(x, on_boundary):
    return on_boundary and x[2] > 0.320 - DOLFIN_EPS


def create_bcs(V, sys_comp, **NS_namespce):
    Q = 5.21E-6   # Re 6500: 6.77E-5 from FDA
    r_0 = 0.006
    u_0 = 2*Q / (r_0*r_0*pi)  # Analytical different r_0
    inn = Expression('u_0 * (1 - (x[0]*x[0] + x[1]*x[1])/r_0/r_0)', u_0=u_0, r_0=r_0)

    bcs = dict((ui, []) for ui in sys_comp)
    bc0 = DirichletBC(V, Constant(0), walls)
    bc10 = DirichletBC(V, inn, inlet)
    bc11 = DirichletBC(V, Constant(0), inlet)
    p2 = DirichletBC(V, Constant(0), outlet)

    bcs['u0'] = [bc0, bc11]
    bcs['u1'] = [bc0, bc11]
    bcs['u2'] = [bc10, bc0]
    bcs['p'] = [p2]

    return bcs


def initialize(q_, **NS_namespace):
    q_['u2'].vector()[:] = 1e-12


def pre_solve_hook(velocity_degree, mesh, pressure_degree, V, nu, **NS_namesepace):
    Vv = VectorFunctionSpace(mesh, 'CG', velocity_degree,
                             constrained_domain=constrained_domain)
    Pv = FunctionSpace(mesh, 'CG', pressure_degree,
                       constrained_domain=constrained_domain)

    r3 = 0.008/0.022685 * 0.006 # On nozzle
    r1 = 0.006
    r2 = 0.002

    z = [-0.18269, -0.088,-0.064, -0.048, -0.02, -0.008, 0.0, \
         0.008, 0.016, 0.024, 0.032, 0.06, 0.08, 0.320]
    radius=[r1, r1, r1, r3, r2, r2, r2, r1, r1, r1, r1, r1, r1, r1]

    slices_u = []
    slices_ss = []
    slices_points = []

    for i in range(len(z)):
        slices_points.append(linspace(-radius[i], radius[i], 1000))
        eval_points = array([[x, 0, z[i]] for x in slices_points[-1]])
        slices_u.append(StatisticsProbes(eval_points.flatten(), Vv))
        slices_ss.append(StatisticsProbes(eval_points.flatten(), Pv))

    # Setup probes in the centerline and at the wall
    z_senterline = linspace(-0.18269, 0.32, 10000)
    x_senter = array([[0.0, 0.0, i] for i in z_senterline])
    x_wall = []

    cone_length = 0.022685

    for z_ in z_senterline:

        # first and last cylinder
        if z_ < -0.62685 or z_ > 0.0:
            r = r1

        # cone
        elif z_ >= -0.62685 and z_ < -0.04:
            r = r1 * (abs(z_) - 0.04) / cone_length

        # narrow cylinder
        elif z_ <= 0.0 and z_ >= -0.04:
            r = r2

        x_wall.append([0.0, r, z_])

    x_wall = array(x_wall)
    senterline_u = StatisticsProbes(x_senter.flatten(), Vv)
    senterline_p = StatisticsProbes(x_senter.flatten(), Pv)
    senterline_ss = StatisticsProbes(x_senter.flatten(), Pv)
    wall_p = StatisticsProbes(x_wall.flatten(), Pv)
    wall_wss = StatisticsProbes(x_wall.flatten(), Pv)
    
    # LagrangeInterpolator for later use
    li = LagrangeInterpolator()
    
    # Box as a basis for a slice
    mesh = BoxMesh(-r1, -r1, -r1, r1, r1, r1, 100, 100, 100)
    bmesh = BoundaryMesh(mesh, "exterior")

    # Create SubMesh for side at z=0
    # This will be a UnitSquareMesh with topology dimension 2 in 3 space
    # dimensions
    cc = CellFunction('size_t', bmesh, 0)
    xyplane = AutoSubDomain(lambda x: x[2] < -r1 + DOLFIN_EPS)
    xyplane.mark(cc, 1)
    submesh = SubMesh(bmesh, cc, 1)

    # Coordinates for the slice
    coordinates = submesh.coordinates()

    # Create a FunctionSpace on the submesh
    Vs = VectorFunctionSpace(submesh, "CG", 1)
    us = Function(Vs)

    # Normal vector
    n = project(Expression(("0", "0", "1")), Vs)

    def flux(u, z):
        # Move slice to z
        coordinates[:, 2] = z

        # LagrangeInterpolator required in parallel
        li.interpolate(us, u)
        
        # Compute flux
        return assemble(dot(us, n)*dx)

    def stress(u):
        def epsilon(u):
            return 0.5*(grad(u) + grad(u).T)
        return project(2*nu*sqrt(inner(epsilon(u),epsilon(u))), Pv)

    return dict(uv=Function(Vv), pv=Function(Pv),# ssv=Functon(Pv), 
                radius=radius, u_diff=Function(Vv), u_prev=Function(Vv), 
                Vv=Vv, wall_p=wall_p, wall_wss=wall_wss, senterline_u=senterline_u,
                senterline_p=senterline_p, senterline_ss=senterline_ss, 
                Pv=Pv, z_senterline=z_senterline, stress=stress,
                slices_u=slices_u, slices_points=slices_points,
                slices_ss=slices_ss, z=z, flux=flux)


def temporal_hook(tstep, info_red, dt, radius, u_diff, pv, stress,
                  Pv, u_prev, u_, check_steady, flux, slices_points,
                  Vv, uv, newfolder, mesh, p_, case, wall_p, wall_wss,
                  z_senterline, folder, senterline_u, senterline_p,
                  senterline_ss, slices_u, slices_ss, z, **NS_namespace):

    # Check steady state
    if tstep % check_steady == 0 \
            and senterline_u.number_of_evaluations() == 0:
        uv.assign(project(u_, Vv))
        file = File(newfolder + "/VTK/nozzle_velocity_%0.2e_%0.2e_%06d.pvd" \
               % (dt, mesh.hmin(), tstep))
        file << uv
        print(flux(uv, -0.1))

        uv.assign(project(u_, Vv))
        u_diff.assign(uv - u_prev)
        diff = norm(u_diff)/norm(uv)
        info_red("Diff: %1.4e   time: %f" % (diff, tstep*dt))

        if diff < 0.5:    #1.5e-2:
            pv.assign(project(p_, Pv))
            ssv = stress(uv)
            
            # Evaluate senterline
            senterline_u(uv)
            senterline_p(pv)
            senterline_ss(ssv)
            
            # Evaluate at the wall
            wall_p(pv)
            wall_wss(ssv)
            
            # Evaluate for each slice
            for i in range(len(slices_u)):
                slices_u[i](uv)
                slices_ss[i](ssv)

        else:
            u_prev.assign(uv)

    if senterline_u.number_of_evaluations() > 0:
        # Variables to store
        uv.assign(project(u_, Vv))
        pv.assign(project(p_, Pv))
        ssv = stress(uv)
        
        # Evaluate senterline
        senterline_u(uv)
        senterline_p(pv)
        senterline_ss(ssv)
        
        # Evaluate at the wall
        wall_p(pv)
        wall_wss(ssv)
        
        # Evaluate for each slice
        for i in range(len(slices_u)):
            slices_u[i](uv)
            slices_ss[i](ssv)

    if senterline_u.number_of_evaluations() == 10:
        file = File(newfolder + "/VTK/nozzle_velocity_%0.2e_%0.2e.pvd"
                    % (dt, mesh.hmin()))
        file << uv
        file = File(newfolder + "/VTK/nozzle_pressure_%0.2e_%0.2e.pvd"
                    % (dt, mesh.hmin()))
        file << pv

        # Save data from the centerline
        info = open(newfolder + "/nozzle_stats.txt", 'w')
        info = write_overhead(info, dt, tstep*dt, mesh.hmin(), case, mesh.hmax())
        
        senterline_u(uv)
        info = write_data(info, senterline_u, z_senterline, "Velocity senterline")

        senterline_ss(ssv)
        info = write_data(info, senterline_ss, z_senterline, "Share stress senterline")

        senterline_p(pv)
        info = write_data(info, senterline_p, z_senterline, "Pressure senterline")
        
        # Save date from the wall
        wall_p(pv)
        info = write_data(info, wall_p, z_senterline, "Wall pressure")

        wall_wss(ssv)
        info = write_data(info, wall_wss, z_senterline, "Wall share stress")

        # Save data from slices
        for i in range(len(slices_u)):
            slices_u[i](uv)
            info = write_data(info, slices_u[i], slices_points[i],
                              "Axial velocity at z=%f" % z[i])

        for i in range(len(slices_ss)):
            slices_ss[i](ssv)
            info = write_data(info, slices_ss[i], slices_points[i], 
                              "Axial share stress at z=%d" % z[i])

        # Saving the flux
        info.write("Flux at each slice\n")
        info.write("%d\n" % (len(z) +2))
        for z_ in [-0.18269] + z + [0.32]:
            info.write("%e %e\n" % (z_, flux(uv, z=z_)))

        info.close()
        kill = open(folder + '/killoasis', 'w')
        kill.close()


def write_overhead(File, dt, T, hmin, Re, hmax):
    File.write("===Nozzle with sudden expanssion===\n")
    File.write("dt=%e\n" % dt)
    File.write("stopp time=%f\n" % T)
    File.write("hmin=%e\n" % hmin)
    File.write("hmax=%s\n" % hmax)
    File.write("Re=%d\n" % Re)
    return File


def write_data(File, probes, points, headline, direction=2):
    array = probes.array() / probes.number_of_evaluations()
    File.write(headline+"\n")
    File.write("%d\n" % len(array))
    if len(array[1]) == 9:
        for i in range(len(array)):
            print(array[i])
            File.write("%e %e\n" % (points[i], array[i][2]))  
    else:
        for i in range(len(array)):
            File.write("%e %e\n" % (points[i], array[i][0]))
    return File
