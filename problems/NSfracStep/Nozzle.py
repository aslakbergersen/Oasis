from ..NSfracStep import *
from math import pi
from os import path, getcwd, listdir, remove, system
from numpy import array, linspace
import sys
from fenicstools import StatisticsProbes
import math
import numpy as np
import cPickle
from mpi4py.MPI import COMM_WORLD as comm
import subprocess

# Values for geometry
start = -0.12
stop = 0.18
r_0 = 0.006
flow_rate = {  # From FDA
             500: 5.21E-6,
             2000: 2.08E-5,
             3500: 3.63E-5,
             5000: 5.21E-5,
             6500: 6.77E-5
            }
inlet_string = 'u_0 * (1 - (x[0]*x[0] + x[1]*x[1])/(r_0*r_0))'
restart_folder = "nozzle_results/data/12/Checkpoint"
#machine_name = subprocess.check_output("hostname", shell=True).split(".")[0]
#nozzle_path = path.sep + path.join("mn", machine_name, "storage", "aslakwb", "nozzle_results")

# Update parameters from last run
if restart_folder is not None:
    restart_folder = path.join(getcwd(), restart_folder)
    f = open(path.join(restart_folder, 'params.dat'), 'r')
    NS_parameters.update(cPickle.load(f))
    NS_parameters['T'] = NS_parameters['T'] + 200 * NS_parameters['dt']
    NS_parameters['restart_folder'] = restart_folder
    globals().update(NS_parameters)
else:
    # Override some problem specific parameters
    recursive_update(NS_parameters,
                    dict(mu=0.0035,
                         rho=1056.,
                         nu=0.0035 / 1056.,
                         T=1e10,
                         dt=5E-5,
                         folder="nozzle_results",
                         case=3500,
                         save_tstep=10,
                         checkpoint=10,
                         check_steady=5,
                         eval_t=1,
                         plot_t=500,
                         velocity_degree=1,
                         pressure_degree=1,
                         mesh_path="mesh/1M_boundary_uniform_nozzle.xml",
                         print_intermediate_info=1000,
                         use_lumping_of_mass_matrix=False,
                         low_memory_version=False,
                         use_krylov_solvers=True,
                         krylov_solvers=dict(monitor_convergence=False,
                                          relative_tolerance=1e-8)))
    

def mesh(mesh_path, **NS_namespace):
    return Mesh(mesh_path)

# This is mesh dependent hmin() / 100 seems to be a good criteria
eps_mesh = 1e-5
def walls(x, on_boundary):
    return on_boundary \
            and ((sqrt(x[0]*x[0] + x[1]*x[1]) > r_0 - eps_mesh) or \
            (x[2] > start + eps_mesh and x[2] < stop - eps_mesh))


def inlet(x, on_boundary):
    return on_boundary and x[2] < start + eps_mesh


def outlet(x, on_boundary):
    return on_boundary and x[2] > stop - eps_mesh


def create_bcs(V, Q, sys_comp, nu, case, mesh, **NS_namespce):
    boundaries = FacetFunction("size_t", mesh)
    boundaries.set_all(0)
    Inlet = AutoSubDomain(inlet)
    Outlet = AutoSubDomain(outlet)
    Walls = AutoSubDomain(walls)
    Walls.mark(boundaries, 1)
    Inlet.mark(boundaries, 2)
    Outlet.mark(boundaries, 3)

    # Compute area of inlet and outlet and adjust radius
    A_walls = assemble(Constant(1)*ds(mesh)[boundaries](1))
    A_in = assemble(Constant(1)*ds(mesh)[boundaries](2))
    A_out = assemble(Constant(1)*ds(mesh)[boundaries](3))

    r_0 = math.sqrt(A_in / math.pi)

    # Find u_0 for 
    u_0 = flow_rate[case] / A_in * 2  # For parabollic inlet
    inn = Expression(inlet_string, u_0=u_0, r_0=r_0)
    no_slip = Constant(0)

    bcs = dict((ui, []) for ui in sys_comp)
    bc0 = DirichletBC(V, no_slip, walls)
    bc10 = DirichletBC(V, inn, inlet)
    bc11 = DirichletBC(V, no_slip, inlet)
    p2 = DirichletBC(Q, no_slip, outlet)

    bcs['u0'] = [bc0, bc11]
    bcs['u1'] = [bc0, bc11]
    bcs['u2'] = [bc0, bc10]
    bcs['p'] = [p2]

    return bcs


def initialize(q_, restart_folder, **NS_namespace):
    if restart_folder is None:
        q_['u2'].vector()[:] = 1e-12


def pre_solve_hook(velocity_degree, mesh, dt, pressure_degree, V,
                   mu, case, newfolder, mesh_path, tstep, **NS_namesepace):

    Vv = VectorFunctionSpace(mesh, 'CG', velocity_degree,
                            constrained_domain=constrained_domain)
    Pv = FunctionSpace(mesh, 'CG', pressure_degree,
                       constrained_domain=constrained_domain)
    DG = FunctionSpace(mesh, 'DG', 0)

    uv = Function(Vv)

    r_2 = 0.008/0.022685 * r_0
    r_1 = r_0 / 3.

    # Location of slices
    z = [-0.088,-0.064, -0.048, -0.02, -0.008, 0.0, \
        0.008, 0.016, 0.024, 0.032, 0.06, 0.08]
    
    # Create a matching list of radius
    radius = []
    for i in range(len(z)):
        if z[i] == -0.048:
            radius.append(r_2)
        elif z[i] <= 0 and z[i] >= -0.02:
            radius.append(r_1)
        else:
            radius.append(r_0)

    # Container for all StatisticsProbes
    eval_dict = {}
    key_u = "slice_u_%s"
    key_ss = "slice_ss_%s"

    eps = 1e-8
    n_slice = 200
    for i in range(len(z)):
        # Set up dict for the slices
        u_ = key_u % z[i]
        ss_ = key_ss % z[i]
        slices_points = linspace(-radius[i]+eps, radius[i]-eps, n_slice)
        points = array([[x, 0, z[i]] for x in slices_points])
        eval_dict[u_] = StatisticsProbes(points.flatten(), Pv, True) 
        eval_dict[ss_] = StatisticsProbes(points.flatten(), Pv, True)

        # Store points
        points.dump(path.join(newfolder, "Stats", "Points", "slice_%s" % z[i]))

    # Setup probes in the centerline and at the wall
    N = 10000
    z_senterline = linspace(start+eps, stop-eps, N)
    eval_senter = array([[0.0, 0.0, i] for i in z_senterline])
    eval_senter.dump(path.join(newfolder, "Stats", "Points", "senterline"))
    eval_wall = []

    cone_length = 0.022685
    for z_ in z_senterline:
        # first and last cylinder
        if z_ < -0.062685 or z_ > 0.0:
            r = r_0 - eps
        # cone
        elif z_ >= -0.062685 and z_ < -0.04:
            r = r_0 * (abs(z_) - 0.04) / cone_length - eps
        # narrow cylinder
        elif z_ <= 0.0 and z_ >= -0.04:
            r = r_1 - eps

        eval_wall.append([0.0, r, z_])

    eval_wall = array(eval_wall)
    eval_wall.dump(path.join(newfolder, "Stats", "Points", "wall"))

    eval_dict["senterline_u"] = StatisticsProbes(eval_senter.flatten(), Pv, True)
    eval_dict["senterline_p"] = StatisticsProbes(eval_senter.flatten(), Pv, True)
    eval_dict["senterline_ss"] = StatisticsProbes(eval_senter.flatten(), Pv, True)
    eval_dict["initial_u"] = StatisticsProbes(eval_senter.flatten(), Pv, True)
    eval_dict["wall_p"] = StatisticsProbes(eval_wall.flatten(), Pv, True)
    eval_dict["wall_ss"] = StatisticsProbes(eval_wall.flatten(), Pv, True)

    if restart_folder is None:
        # Print header
        if MPI.rank(mpi_comm_world()) == 0:
            print_header(dt, mesh.hmax(), mesh.hmin(), case, start, stop,
                         inlet_string, mesh.num_cells(), newfolder, mesh_path)

    else:
        # Restart stats
        files = listdir(path.join(newfolder, "Stats"))
        eval = int(files[0].split("_")[-1])
        if files != [] and eval > 0:
            for file in files:
                if file == "Points" or file == "initial_u_%s" % eval: 
                    continue
                file_split = file.split("_")
                key = "_".join(file_split[:-1])
                arr = np.load(path.join(newfolder, "Stats", file))
                eval_dict[key].restart_probes(arr.flatten(), eval)

	    if tstep*dt > 0.000002:
		eval_dict.pop("initial_u")
        
        else:
            if MPI.rank(mpi_comm_world()) == 0:
                print "WARNING:  The stats folder is empty and the stats is not restarted"

    # For length scale
    h = CellVolume(mesh)
    dl = project(12/math.sqrt(2) * h**(1./3), DG)
    l_pluss = Function(DG)
    t_pluss = Function(DG)

    # For stress eval
    v = TestFunction(DG)
    ssv = Function(DG)
    def stress(u):
        def epsilon(u):
            return 0.5*(grad(u) + grad(u).T)
        f = 2*mu*sqrt(inner(epsilon(u),epsilon(u)))
        x = assemble(inner(f, v)/h*dx(mesh))
        ssv.vector().set_local(x.array())
        ssv.vector().apply("insert")
        return ssv
    
    def norm_l(u, l=2):
        if l == "max":
            return np.max(abs(u.flatten()))
        else:
            return np.sum(u**l)**(1./l)

    # Files to store plot
    file_u = File(path.join(newfolder, "VTK", "velocity.pvd"))
    file_p = File(path.join(newfolder, "VTK", "pressure.pvd"))
    file_ss = File(path.join(newfolder, "VTK", "stress.pvd"))
    file_l = File(path.join(newfolder, "VTK", "length_scale.pvd"))
    file_t = File(path.join(newfolder, "VTK", "time_scale.pvd"))
    file_v = File(path.join(newfolder, "VTK", "velocity_scale.pvd"))
    files = {"u": file_u, "p": file_p, "ss": file_ss,
             "t": file_t, "l": file_l, "v": file_v}

    # For flux evaluation in inlet, outlet and walls
    normal = FacetNormal(mesh)
    Inlet = AutoSubDomain(inlet)
    Outlet = AutoSubDomain(outlet)
    Walls = AutoSubDomain(walls)
    domains = FacetFunction('size_t', mesh, 0)
    Inlet.mark(domains, 1)
    Outlet.mark(domains, 2)
    Walls.mark(domains, 3)

    # For stopping criteria
    prev = [zeros((N, 3))]

    return dict(Vv=Vv, Pv=Pv, DG=DG, z=z, files=files, stress=stress, prev=prev,
                norm_l=norm_l, eval_dict=eval_dict, normal=normal, domains=domains, 
                dl=dl, l_pluss=l_pluss, t_pluss=t_pluss, uv=uv)
    

def temporal_hook(u_, p_, newfolder, mesh, check_steady, Vv, Pv, tstep, eval_dict, 
                  norm_l, nu, z, rho, DG, eval_t, files, T, folder, stress, prev,
                  normal, dt, domains, plot_t, checkpoint, dl, t_pluss, l_pluss,
                  uv, **NS_namespace):

    # Print timestep
    if tstep % eval_t == 0:
        if MPI.rank(mpi_comm_world()) == 0:
            print tstep

    if tstep % check_steady == 0 and eval_dict.has_key("initial_u"): 
        # Store vtk files for post prosess in paraview 
        [assign(uv.sub(i), u_[i]) for i in range(mesh.geometry().dim())]
        files["u"] << uv

        inlet_flux = assemble(dot(uv, normal)*ds(mesh)[domains](1))
        outlet_flux = assemble(dot(uv, normal)*ds(mesh)[domains](2))
        walls_flux = assemble(dot(uv, normal)*ds(mesh)[domains](3))

        if MPI.rank(mpi_comm_world()) == 0:
            print "Flux in: %e out: %e walls:%e" % (inlet_flux, outlet_flux, walls_flux)

        # Initial conditions is "washed away"
        if tstep*dt > 0.0000002:
            if MPI.rank(mpi_comm_world()) == 0:
                print "="*25 + "\n DONE WITH FIRST ROUND\n\t%s\n" % tstep + "="*25
            eval_dict.pop("initial_u")
    
    if not eval_dict.has_key("initial_u"):
        # Evaluate points
        ssv = stress(as_vector(u_))
        evaluate_points(eval_dict, {"u": u_, "p": p_, "ss": ssv})

        if tstep % plot_t == 0:
            # Compute scales for mesh evaluation
            [assign(uv.sub(i), u_[i]) for i in range(mesh.geometry().dim())]

            u_star = ssv.vector().array() / (2 * rho)

            l_pluss.vector().set_local(np.sqrt(u_star) * dl.vector().array() / nu)
            l_pluss.vector().apply("insert")

            t_pluss.vector().set_local(nu / u_star)
            t_pluss.vector().apply("insert")

            l_pluss.rename("l+", "length scale")
            t_pluss.rename("t+", "time scale")
            ssv.rename("Shear stress", "Shear stress")
            uv.rename("u", "velocity")
            p_.rename("p", "pressure")

            # Store vtk files for post process in paraview 
            t_ = T * tstep
            files["u"] << uv, t_
            files["l"] << l_pluss, t_
            files["t"] << t_pluss, t_
            files["p"] << p_, t_
            files["ss"] << ssv, t_

        if tstep % check_steady == 0:
            # Check the max norm of the difference
            num = eval_dict["senterline_u"].number_of_evaluations()
            arr = eval_dict["senterline_u"].array()
            arr = comm.bcast(arr, root=0)  # Might be better to do bcast after norm_l
            arr_ = arr[:,:3] / num - prev[0]

            norm = norm_l(arr_, l="max")
		
            # Update prev 
            prev[0] = (arr[:,:3] / num).copy()

            # Print info
            if MPI.rank(mpi_comm_world()) == 0:
                print "Condition:", norm < 0.00001,
                print "On timestep:", tstep,
                print "Norm:", norm

            # Check if stats have stabilized
            if norm < 0.00001:
                dump_stats(eval_dict, newfolder)

                # Clean kill of program
                if MPI.rank(mpi_comm_world()) == 0:
                    kill = open(folder + '/killoasis', 'w')
                    kill.close()
                MPI.barrier(mpi_comm_world())

    if tstep % checkpoint == 0:
        dump_stats(eval_dict, newfolder)


def dump_stats(eval_dict, newfolder):
    filepath = path.join(newfolder, "Stats")
    
    # Remove previous stats files
    if MPI.rank(mpi_comm_world()) == 0:
        if listdir(filepath) != []:
            for file in listdir(filepath):
                if path.isfile(path.join(filepath, file)):
                    remove_path = path.join(filepath, file)
                    remove(remove_path)
    MPI.barrier(mpi_comm_world())

    # Dump stats, store number of evaluations in filename
    for key, value in eval_dict.iteritems():
        arr = value.array()
        if MPI.rank(mpi_comm_world()) == 0:
            arr = arr / value.number_of_evaluations()
            arr.dump(path.join(filepath, key + "_" + str(value.number_of_evaluations())))


def evaluate_points(eval_dict, eval_map):
    for key, value in list(eval_dict.iteritems()):
        k = key.split("_")[1]
        sample = eval_map[key.split("_")[1]]
        if k == "u":
            # Segregated probe eval
            value(sample[0], sample[1], sample[2])
        else:
            value(sample)


def print_header(dt, hmin, hmax, Re, start, stopp, inlet_string, 
                 num_cell, folder, mesh_path):
    file = open(path.join(folder, "problem_paramters.txt"), "w")
    file.write("=== Nozzle with sudden expanssion ===\n")
    file.write("dt=%e\n" % dt)
    file.write("hmin=%e\n" % hmin)
    file.write("hmax=%s\n" % hmax)
    file.write("Re=%d\n" % Re)
    file.write("Start=%s\n" % start)
    file.write("Stopp=%s\n" % stopp)
    file.write("Inlet=%s\n" % inlet_string)
    file.write("Number of cells=%s\n" % num_cell)
    file.write("Path to mesh=%s\n" % mesh_path)
    file.close()
