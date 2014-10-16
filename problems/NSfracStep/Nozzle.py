from dolfin import *
from math import pi
from fenicstools import StatisticsProbes    # StructuredGrid,
from numpy import array, linspace           # sum

# Override some problem specific parameters
recursive_update(NS_parameters,
                 dict(n=0.0035 / 1056.,
                      T=1000,
                      dt=0.0001,
                      folder="nozzle_results",
                      case=500,
                      steady=False,
                      save_tstep=1000,
                      checkpoint=1000,
                      check_steady=100,
                      velocity_degree=1,
                      mesh_path="mesh/8M_nozzle.xml",
                      print_intermediate_info=1000,
                      use_lumping_of_mass_matrix=False,
                      low_memory_version=False,
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
    Q = 5.21E-6   # Re 6500: 6.77E-5  # From FDA
    r_0 = 0.006
    u_maks = Q / (4*r_0*r_0*(1-2/pi))  # Analytical different r_0
    inn = Expression('u_maks * cos(sqrt(pow(x[0],2) + \
            pow(x[1],2))/r_0/2.*pi)', u_maks=u_maks, r_0=r_0)

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


def pre_solve_hook(velocity_degree, mesh, pressure_degree, V, **NS_namespace):
    # To compute uv and flux
    Vv = VectorFunctionSpace(mesh, 'CG', velocity_degree,
                             constrained_domain=constrained_domain)
    Pv = FunctionSpace(mesh, 'CG', pressure_degree,
                       constrained_domain=constrained_domain)
    normal = FacetNormal(mesh)

    # No need to make the entire slice for validation.
    # only need to avaluate in a line at this point
    # And since the geometry is relativly small
    # (for now) it's ok to store the entire
    # figure for vizualisation.
    # Create 12 slices
    # origin = [-0.006, -0.006, 0.0]
    # vectors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    # dL = [0.012, 0.012, 0]
    # N = [200, 200, 1]
    # r3 = 0.008/0.022685 * 0.006 # On nozzle
    # r1 = 0.006
    # r2 = 0.002

    # z = [-0.18269, -0.088,-0.064, -0.048, -0.02, -0.008, 0.0, \
    #         0.008, 0.016, 0.024, 0.032, 0.06, 0.08, 0.320]
    # radius=[r1, r1, r1, r3, r2, r2, r2, r1, r1, r1, r1, r1, r1, r1]
    # stats=[]

    # for z_ in z:
    #     stats.append(StructuredGrid(V, N, [-0.006, -0.006, z_],
    #                                  vectors, dL, statistics=True))

    # Normals and domains to compute flux at at
    # each point (z), inlet and outlet
    normal = FacetNormal(mesh)
    Inlet = AutoSubDomain(inlet)
    Outlet = AutoSubDomain(outlet)
    domains = FacetFunction('size_t', mesh, 0)

    # mark domanis
    Inlet.mark(domains, 1)
    Outlet.mark(domains, 2)

    # Evaluate the centerline and the wall
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
    senterline = StatisticsProbes(x_senter.flatten(), V)
    wall = StatisticsProbes(x_wall.flatten(), V)

    return dict(uv=Function(Vv), pv=Function(Pv), radius=radius,
                u_diff=Function(Vv), u_prev=Function(Vv), Vv=Vv,
                wall=wall, senterline=senterline, domains=domains,
                Pv=Pv, z_senterline=z_senterline, normal=normal, stats=stats)


def temporal_hook(tstep, info_red, steady, dt, radius, u_diff, pv,
                  Pv, u_prev, u_, check_steady, domains, normal,
                  Vv, uv, newfolder, mesh_resolution, mesh, p_, case,
                  z_senterline, wall, folder, senterline, stats,
                  **NS_namespace):

    # steady state
    if tstep % check_steady == 0:
        # uv.assign(project(u_, Vv))
        # file = File(newfolder + "/VTK/nozzle_velocity_%0.2e_%0.2e_%d.pvd" \
        #        % (dt, mesh.hmin(), tstep))
        # file << uv

        uv.assign(project(u_, Vv))
        u_diff.assign(uv - u_prev)
        diff = norm(u_diff)/norm(uv)
        print "Diff: %1.4e   time: %f" % (diff, tstep*dt)

        inlet_flux = assemble(dot(u_, normal)*ds(1),
                              exterior_facet_domains=domains)
        outlet_flux = assemble(dot(u_, normal)*ds(2),
                               exterior_facet_domains=domains)
        rel_err = (abs(inlet_flux) - abs(outlet_flux)) / abs(inlet_flux)
        # TODO: change with theoretical value

        if MPI.process_number() == 0:
            info_red("Flux in: %e\nFlux out: %e\nRelativ error: %e\ntstep: %d"
                     % (inlet_flux, outlet_flux, rel_err, tstep))

        if diff < 0.5:  # 2.5e-4:
            steady = True   # Dave stats and kill the program
        else:
            u_prev.assign(uv)

    if steady:
        print 'OK'
        # save a plot
        uv.assign(project(u_, Vv))
        pv.assign(project(p_, Pv))
        print 'OK1'
        senterline(uv)
        print 'OK2'
        wall(pv)
        print 'OK3'
        for i in range(len(stats)):
            stats[i](uv)
        print 'OK4'

    if senterline.number_of_evaluations == 100:
        file = File(newfolder + "/VTK/nozzle_velocity_%0.2e_%0.2e.pvd"
                    % (dt, mesh.hmin()))
        file << uv
        file = File(newfolder + "/VTK/nozzle_pressure_%0.2e_%0.2e.pvd"
                    % (dt, mesh.hmin()))
        file << pv

        # save the data from the centerline
        info = open(newfolder + "/Stats/nozzle_stats.txt", 'w')
        info = write_overhead(info, dt, tstep*dt, mesh.hmin(),
                              case, mesh_resolution)
        info.write("Axial velocity\n")
        info.write("   z            value\n")

        senterline(uv)
        data = senterline.array()
        for i in range(len(data)):
            info.write("%f %s\n" % (z_senterline[i], data[i]))

        # save information for each slice
        for i in range(len(stats)):
            stats[i](uv)
            stats[i].toh5(0, tstep, filename=newfolder+"/VTK/slice_z_%s.h5"
                          % (radius[i]))

        # save preassure
        info.write("Wall pressure\n")

        wall(pv0)
        data = wall.array()
        info.write("   z            value\n")
        for i in range(len(data)):
            info.write("%f %s\n" % (z_senterline[i], data[i]))

        info.close
        kill = open(folder + '/killoasis', 'w')
        kill.close
        # TODO: create a file that is called killoasis?


def write_overhead(File, dt, T, hmin, Re, mesh_resolution):
    File.write("===Nozzle with sudden expanssion===\n")
    File.write("dt=%e\n" % dt)
    File.write("stopp time=%f\n" % T)
    File.write("hmin=%e\n" % hmin)
    File.write("Re=%d\n" % Re)
    File.write("Mesh=%s\n" % mesh_resolution)
    return File
