from ..NSfracStep import *
from math import pi
from os import path, getcwd, listdir, remove, system, makedirs
from numpy import array, linspace
import sys
from fenicstools import StatisticsProbes, Probes
import math
import numpy as np
import cPickle
from mpi4py.MPI import COMM_WORLD as comm
import subprocess

# Values for geometry
start = -0.12
stop = 0.20
r_0 = 0.006
flow_rate = {  # From FDA
             500: 5.21E-6,
             2000: 2.08E-5,
             3500: 3.63E-5,
             5000: 5.21E-5,
             6500: 6.77E-5
            }
inlet_string = 'u_0 * (1 - (x[0]*x[0] + x[1]*x[1])/(r_0*r_0))'
#restart_folder = None #"nozzle_results/data/12/Checkpoint"
#machine_name = subprocess.check_output("hostname", shell=True).split(".")[0]
#nozzle_path = path.sep + path.join("mn", machine_name, "storage", "aslakwb", "nozzle_results")

# Update parameters from last run
def update(commandline_kwargs, NS_parameters, **NS_namespace):
    if commandline_kwargs.has_key("restart_folder"):
        restart_folder = commandline_kwargs["restart_folder"]
        restart_folder = path.join(getcwd(), restart_folder)
        f = open(path.join(path.dirname(path.abspath(__file__)), restart_folder, 'params.dat'), 'r')
        NS_parameters.update(cPickle.load(f))
        NS_parameters['T'] = NS_parameters['T'] + 200 * NS_parameters['dt']
        NS_parameters['restart_folder'] = restart_folder
        globals().update(NS_parameters)
    else:
        # Override some problem specific parameters
        d = recursive_update(NS_parameters,
                        dict(mu=0.0035,
                            rho=1056.,
                            nu=0.0035 / 1056.,
                            T=1e10,
                            dt=5E-5,
                            folder="nozzle_results",
                            case=3500,
                            save_tstep=10e10,
                            checkpoint=1000,
                            check_steady=300,
                            eval_t=50,
                            plot_t=10,
                            velocity_degree=1,
                            pressure_degree=1,
                            mesh_path="mesh/course_mesh_for_Oasis.xml.gz",
                            print_intermediate_info=1000,
                            use_lumping_of_mass_matrix=False,
                            low_memory_version=False,
                            use_krylov_solvers=True,
                            krylov_solvers=dict(monitor_convergence=False,
                                            relative_tolerance=1e-8)))
        globals().update(d)
    

def mesh(mesh_path, **NS_namespace):
    return Mesh(mesh_path)

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


def initialize(q_, restart_folder, mesh_path, **NS_namespace):
    if restart_folder is None:
        q_['u2'].vector()[:] = 1e-12


def pre_solve_hook(velocity_degree, mesh, dt, pressure_degree, V,
                   mu, case, newfolder, mesh_path, tstep, **NS_namesepace):

    MPI.barrier(mpi_comm_world())
    if MPI.rank(mpi_comm_world()) == 0:
        if not path.isdir(path.join(newfolder, "Stats", "Probes")):
            makedirs(path.join(newfolder, "Stats", "Probes"))
        if not path.isdir(path.join(newfolder, "Stats", "Points")):
            makedirs(path.join(newfolder, "Stats", "Points"))

    Vv = VectorFunctionSpace(mesh, 'CG', velocity_degree,
                            constrained_domain=constrained_domain)
    Pv = FunctionSpace(mesh, 'CG', pressure_degree,
                       constrained_domain=constrained_domain)
    DG = FunctionSpace(mesh, 'DG', 0)

    uv = Function(Vv)

    length_cone = 0.022685
    r_1 = r_0 / 3.
    r_2 = r_1 + ((length_cone - 0.008)/length_cone * (r_0 - r_1))

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

    eps = 1e-8
    n_slice = 200
    for i in range(len(z)):
        # Set up dict for the slices
        u_ = key_u % z[i]
        slices_points = linspace(-radius[i], radius[i], n_slice)
        points = array([[x, 0, z[i]] for x in slices_points])
        eval_dict[u_] = StatisticsProbes(points.flatten(), Pv, True) 

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
            r = r_1 + (abs(z_) - 0.04) / cone_length * (r_0 - r_1) - eps

        # narrow cylinder
        elif z_ <= 0.0 and z_ >= -0.04:
            r = r_1 - eps

        eval_wall.append([0.0, r, z_])

    eval_wall = array(eval_wall)
    eval_wall.dump(path.join(newfolder, "Stats", "Points", "wall"))

    # Make probe points
    probe_list = [-25] + range(-15, 0, 2) + range(16) + [20, 30, 40]
    probe_points = []
    for j in range(2, -3, -1):
    	probe_points += [[r_1*j, 0, r_1*2*i] for i in probe_list]
    probe_points = array(probe_points)
    probe_points.dump(path.join(newfolder, "Stats", "Probes", "points"))
    
    eval_dict["senterline_u"] = StatisticsProbes(eval_senter.flatten(), Pv, True)
    eval_dict["senterline_p"] = StatisticsProbes(eval_senter.flatten(), Pv, True)
    eval_dict["initial_u"] = StatisticsProbes(eval_senter.flatten(), Pv, True)
    eval_dict["wall_p"] = StatisticsProbes(eval_wall.flatten(), Pv, True)
    eval_dict["senterline_u_probes"] = Probes(probe_points.flatten(), Vv)
    eval_dict["senterline_p_probes"] = Probes(probe_points.flatten(), Pv)

    # Finding the mean velocity
    u_mean = {"u": Function(Vv), "num": 0}

    if restart_folder is None:
        # Print header
        if MPI.rank(mpi_comm_world()) == 0:
            print_header(dt, mesh.hmax(), mesh.hmin(), case, start, stop,
                         inlet_string, mesh.num_cells(), newfolder, mesh_path)

    else:
        # Restart stats
        files = listdir(path.join(newfolder, "Stats"))
        files = [f for f in files if path.isfile(path.join(newfolder, "Stats", f))]
        files.sort()
        eval = 0 if len(files) == 0 else int(files[-1].split("_")[-1])
        if files != [] and eval > 0:
            for file in files:
                if path.isdir(path.join(newfolder, "Stats", file)) or file == "initial_u_%s" % eval: 
                    continue
                file_split = file.split("_")
                key = "_".join(file_split[:-1])
                arr = np.load(path.join(newfolder, "Stats", file))
                eval_dict[key].restart_probes(arr.flatten(), eval)

	    if tstep*dt > 0.4:
		eval_dict.pop("initial_u")
        else:
            if MPI.rank(mpi_comm_world()) == 0:
                print "WARNING: The stats folder is empty and the stats is not restarted"

        # Restart mean velocity
        files = listdir(path.join(newfolder, "VTK"))
        files = [path.join(newfolder, "VTK", f) for f in files if "u_mean_num" in f]
        if files != []:
            file = sorted(files, key= lambda x: int(x.split("_")[2][3:]))[0]
            u_mean["num"] = int(file.split("_")[2][3:])
            tmp = Function(Vv, file)
            u_mean["u"].vector().axpy(u_mean["num"], tmp.vector()) 

    def norm_l(u, l=2):
        if l == "max":
            return np.max(abs(u.flatten()))
        else:
            return np.sum(u**l)**(1./l)

    # Files to store plot
    file_u = path.join(newfolder, "VTK", "velocity%06.0f.xml.gz")
    file_p = path.join(newfolder, "VTK", "pressure%06.0f.xml.gz")
    files = {"u": file_u, "p": file_p}

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

    return dict(Vv=Vv, Pv=Pv, DG=DG, z=z, files=files, prev=prev,
                norm_l=norm_l, eval_dict=eval_dict, normal=normal, domains=domains, 
                uv=uv, u_mean=u_mean)


def temporal_hook(u_, p_, newfolder, mesh, check_steady, Vv, Pv, tstep, eval_dict, 
                  norm_l, nu, z, rho, DG, eval_t, files, T, folder, prev, u_mean,
                  normal, dt, domains, plot_t, checkpoint, uv, **NS_namespace):
    # Print timestep
    if tstep % eval_t == 0:
        if MPI.rank(mpi_comm_world()) == 0:
            print tstep

    if tstep % check_steady == 0 and eval_dict.has_key("initial_u"):
	[assign(uv.sub(i), u_[i]) for i in range(mesh.geometry().dim())] 
        inlet_flux = assemble(dot(uv, normal)*ds(mesh)[domains](1))
        outlet_flux = assemble(dot(uv, normal)*ds(mesh)[domains](2))
        walls_flux = assemble(dot(uv, normal)*ds(mesh)[domains](3))

        if MPI.rank(mpi_comm_world()) == 0:
            print "Flux in: %e out: %e walls:%e" % (inlet_flux, outlet_flux, walls_flux)

        # Initial conditions is "washed away"
        if tstep*dt > 0.4:
            if MPI.rank(mpi_comm_world()) == 0:
                print "="*25 + "\n DONE WITH FIRST ROUND\n\t%s\n" % tstep + "="*25
            eval_dict.pop("initial_u")
    
    if not eval_dict.has_key("initial_u"):
        # Evaluate points
        [assign(uv.sub(i), u_[i]) for i in range(mesh.geometry().dim())]
        evaluate_points(eval_dict, {"u": u_, "p": p_}, uv)
        u_mean["u"].vector()[:] += uv.vector()
        u_mean["num"] += 1 

        if tstep % plot_t == 0:
            uv.rename("u", "velocity")
            p_.rename("p", "pressure")

            # Store vtk files for post process in paraview 
            t_ = T * tstep
            components = {"u": uv, "p": p_}
            for key in components.keys():
	        file = File(files[key] % tstep)
                file << components[key], t_

        
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
                dump_stats(eval_dict, newfolder, dt, tstep)

                # Clean kill of program
                if MPI.rank(mpi_comm_world()) == 0:
                    kill = open(folder + '/killoasis', 'w')
                    kill.close()
                MPI.barrier(mpi_comm_world())

        if tstep % checkpoint == 0 and not eval_dict.has_key("initial_u"):
            dump_stats(eval_dict, newfolder, dt, tstep)
            dump_mean(u_mean, newfolder, tstep, dt, Vv)


def dump_mean(u_mean, newfolder, tstep, dt, Vv):
     filepath = path.join(newfolder, "VTK")

     # Back up u_mean
     file = File(path.join(filepath, "u_mean_num%s_tstep%s_dt%s.xml.gz" % \
                                              (u_mean["num"], tstep, dt)))
     tmp = Function(Vv)
     tmp.vector().axpy(1. / u_mean["num"], u_mean["u"].vector())
     file << tmp


def dump_stats(eval_dict, newfolder, dt, tstep):
    filepath = path.join(newfolder, "Stats")

    # Dump stats, store number of evaluations in filename
    for key, value in eval_dict.iteritems():
        if key.split("_")[-1] != "probes":
            arr = value.array()
        else:
            f_name = path.join(filepath, "Probes", key.split("_")[1]+"_"+str(dt)+"_"+str(tstep))
            value.array(filename=f_name)
            value.clear()
        if MPI.rank(mpi_comm_world()) == 0:
            if key.split("_")[-1] == "probes":
                continue
            arr = arr / value.number_of_evaluations()
            arr.dump(path.join(filepath, key + "_" + str(value.number_of_evaluations())))

    # Remove previous stats files
    num_eval = str(value.number_of_evaluations())
    if MPI.rank(mpi_comm_world()) == 0:
        if listdir(filepath) != []:
            for file in listdir(filepath):
                if path.isfile(path.join(filepath, file)) and not num_eval in file:
                    remove_path = path.join(filepath, file)
                    remove(remove_path)
    MPI.barrier(mpi_comm_world())


def evaluate_points(eval_dict, eval_map, uv):
    for key, value in list(eval_dict.iteritems()):
        k = key.split("_")[1]
        sample = eval_map[key.split("_")[1]]
        if k == "u" and key.split("_")[-1] != "probes":
            # Segregated probe eval
            value(sample[0], sample[1], sample[2])
        elif k == "u":
            value(uv)
        else:
            value(sample)


def print_header(dt, hmin, hmax, Re, start, stopp, inlet_string, 
                 num_cell, folder, mesh_path):
    file = open(path.join(folder, "problem_parameters.txt"), "w")
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
