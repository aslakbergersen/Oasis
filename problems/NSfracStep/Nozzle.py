from ..NSfracStep import *
from math import pi
from os import path, getcwd, mkdir, listdir, remove
from fenicstools import StatisticsProbes
from numpy import array, linspace
import sys
import numpy as np
import cPickle
from mpi4py.MPI import COMM_WORLD as comm

# Values for geometry
start = -0.18
stop = 0.12
r_0 = 0.006
flow_rate = {  # From FDA
             500: 5.21E-6,
             2000: 2.08E-5,
             3500: 3.63E-5,
             5000: 5.21E-5,
             6500: 6.77E-5
            }
inlet_string = 'u_0 * (1 - (x[0]*x[0] + x[1]*x[1])/(r_0*r_0))'
restart_folder = None #"nozzle_results/data/88/Checkpoint"

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
                        T=1000,
                        dt=4E-5,
                        folder="nozzle_results",
                        case=3500,
                        save_tstep=1000,
                        checkpoint=1,
                        check_steady=100,
                        velocity_degree=1,
                        pressure_degree=1,
                        mesh_path="mesh/1600K_opt_nozzle.xml",
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
        and x[2] < stop - DOLFIN_EPS \
        and x[2] > start + DOLFIN_EPS


# inlet = 1
def inlet(x, on_boundary):
    return on_boundary and x[2] < start + DOLFIN_EPS


# outlet = 2
def outlet(x, on_boundary):
    return on_boundary and x[2] > stop - DOLFIN_EPS


def create_bcs(V, sys_comp, case, **NS_namespce):
    u_0 = 2*flow_rate[case] / (r_0*r_0*pi)   # Need to find mesh inlet area
    inn = Expression(inlet_string, u_0=u_0, r_0=r_0)

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


def initialize(q_, restart_folder, **NS_namespace):
    if restart_folder is None:
        q_['u2'].vector()[:] = 1e-12


def pre_solve_hook(velocity_degree, mesh, dt, pressure_degree, V, mu, case,
                folder, newfolder, mesh_path, **NS_namesepace):
    Vv = VectorFunctionSpace(mesh, 'CG', velocity_degree,
                            constrained_domain=constrained_domain)
    Pv = FunctionSpace(mesh, 'CG', pressure_degree,
                       constrained_domain=constrained_domain)
    DG = FunctionSpace(mesh, 'DG', 0)

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

    # List with probes at each slice
    slices_u = []
    slices_ss = []

    for i in range(len(z)):
        slices_points = linspace(-radius[i], radius[i], 200)
        eval_points = array([[x, 0, z[i]] for x in slices_points])
        slices_u.append([StatisticsProbes(eval_points.flatten(), Vv, False), z[i]])
        slices_ss.append([StatisticsProbes(eval_points.flatten(), Pv, True), z[i]])

    # Setup probes in the centerline and at the wall
    z_senterline = linspace(start, stop, 10000)
    eval_senter = array([[0.0, 0.0, i] for i in z_senterline])
    eval_wall = []

    cone_length = 0.022685

    for z_ in z_senterline:
        # first and last cylinder
        if z_ < -0.62685 or z_ > 0.0:
            r = r_0
        # cone
        elif z_ >= -0.62685 and z_ < -0.04:
            r = r_0 * (abs(z_) - 0.04) / cone_length
        # narrow cylinder
        elif z_ <= 0.0 and z_ >= -0.04:
            r = r_1

        eval_wall.append([0.0, r, z_])

    eval_wall = array(eval_wall)

    senterline_u = StatisticsProbes(eval_senter.flatten(), Vv, False)
    senterline_p = StatisticsProbes(eval_senter.flatten(), Pv, True)
    senterline_ss = StatisticsProbes(eval_senter.flatten(), Pv, True)
    initial_u = StatisticsProbes(eval_senter.flatten(), Vv, False)
    wall_p = StatisticsProbes(eval_wall.flatten(), Pv, True)
    wall_wss = StatisticsProbes(eval_wall.flatten(), Pv, True)

    # Gather all probes in a dict
    eval_dict = {"wall_p": wall_p,
                "wall_ss": wall_wss,
                "senterline_ss": senterline_ss,
                "senterline_p": senterline_p,
                "senterline_u": senterline_u,
                "initial_u": initial_u,
                "slices_u": slices_u,
                "slices_ss": slices_ss}

    if restart_folder is None:
        if MPI.rank(mpi_comm_world()) == 0:
            # Print header
            u_0 = 2*flow_rate[case] / (r_0*r_0*pi)
            print_header(dt, mesh.hmax(), mesh.hmin(), case, start, stop, u_0,
                         inlet_string, mesh.num_cells(), folder, mesh_path)

            # Create stats folder
            mkdir(path.join(newfolder, "Stats"))
   
    else:
        # Restart stats
        files = listdir(path.join(newfolder, "Stats"))
        if files != []:
            for file in files:
                file_split = file.split("_")
                key = file_split[0] + "_" + file_split[1]
                num_eval = int(file_split[-2])
                file_path = path.join(newfolder, "Stats", file)
                arr = np.load(file_path)
                if "slices" in file:
                    file_split = file.split("_")
                    z_point = float(file_split[2])
                    index = z.index(z_point)
                    eval_dict[key][index][0].restart_probes(arr.flatten(), num_eval)
                else:
                    eval_dict[key].restart_probes(arr.flatten(), num_eval)

    def stress(u):
        def epsilon(u):
            return 0.5*(grad(u) + grad(u).T)
        return project(2*mu*sqrt(inner(epsilon(u),epsilon(u))), DG)
    
    def norm_l(u, scale, l=2):
        # Runs in series, so np. instead of MPI. ?
        if MPI.rank(mpi_comm_world()) == 0:
            u = u / scale
            if l == "max":
                return np.max(abs(u))
            else:
                return np.sum(u**l)**(1./l)
        else:
            return 0
    
    uv=Function(Vv)
    pv=Function(Pv)
    eval_map = {"p": pv, "ss": stress, "u": uv}
        
    return dict(Vv=Vv, Pv=Pv, eval_map=eval_map,DG=DG,
                norm_l=norm_l, eval_dict=eval_dict)
    
    
def temporal_hook(u_, p_, newfolder, mesh, folder, check_steady, Vv, Pv, tstep, eval_dict, 
                norm_l, eval_map, dt, checkpoint, nu, mu, DG, **NS_namespace):

    if (tstep % check_steady == 0 and \
        eval_dict["initial_u"].number_of_evaluations() != 0) or \
        tstep == 1:
        
        # Compare the norm of the stats
        initial_u = eval_dict["initial_u"]
        num = initial_u.number_of_evaluations()
        bonus = 1 if num == 0 else 0

        arr = initial_u.array()
        prev_norm = norm_l(arr, num + bonus)
        initial_u(u_[0], u_[1], u_[2])
        arr = initial_u.array()
        new_norm = norm_l(arr, num + 1)

        prev_norm = comm.bcast(prev_norm, root=0)
        new_norm = comm.bcast(new_norm, root=0)
        # Max norm is perhaps better suited?
        if MPI.rank(mpi_comm_world()) == 0:
            print "Condition:", abs(prev_norm - new_norm) / new_norm < 0.001,
            print "On timestep:", tstep,
            print "New norm:", new_norm,
            print "Prev norm:", prev_norm,
            print "Relativ diff:", abs(prev_norm - new_norm) / new_norm
        if abs(prev_norm - new_norm) / new_norm < 0.001:
            if MPI.rank(mpi_comm_world()) == 0:
                print "="*25 + "\n DONE WITH FIRST ROUND\n" + "="*25
            eval_dict["initial_u"].clear()
        
    if eval_dict["initial_u"].number_of_evaluations() == 0:
        eval_map["u"].assign(project(u_, Vv))
        eval_map["p"].assign(project(p_, Pv))
        ssv = eval_map["ss"](eval_map["u"])
        file = File(newfolder + "/VTK/nozzle_velocity_%0.2e_%0.2e_%06d.pvd" \
                    % (dt, mesh.hmin(), tstep))
        file << eval_map["u"]

        #plot(ssv, interactive=True)

        num = eval_dict["senterline_u"].number_of_evaluations()
        bonus = 1 if num == 0 else 0
        arr = eval_dict["senterline_u"].array()
        prev_norm = norm_l(arr, num + bonus)

        for key, value in eval_dict.iteritems():
            sample = eval_map[key.split("_")[-1]]
            sample = sample if not type(sample) == type(lambda x: 1) else ssv
            if "slices" in key:
                for val in value:
                    val[0](sample)
            elif key != "initial_u":
                value(sample)
        
        arr = eval_dict["senterline_u"].array()
        new_norm = norm_l(arr, num+1)

        prev_norm = comm.bcast(prev_norm, root=0)
        new_norm = comm.bcast(new_norm, root=0)

        #TODO: What about the (<ee>) mean?
        nu = Constant(nu)
        mu = Constant(mu)
        
        epsilon = project(ssv / (2*mu) * sqrt(nu*2), DG)
        
        time_scale = project(sqrt(sqrt(nu) * epsilon), DG)
        length_scale = project(sqrt(sqrt(nu**3) / epsilon), DG)
        velocity_scale = project(sqrt(nu) / epsilon, DG)
 
        file = File(newfolder + "/VTK/nozzle_length_scale_%0.2e_%0.2e_%06d.pvd" \
                    % (dt, mesh.hmin(), tstep))
        file << length_scale

        file = File(newfolder + "/VTK/nozzle_time_scale_%0.2e_%0.2e_%06d.pvd" \
                    % (dt, mesh.hmin(), tstep))
        file << time_scale

        file = File(newfolder + "/VTK/nozzle_velocity_scale_%0.2e_%0.2e_%06d.pvd" \
                    % (dt, mesh.hmin(), tstep))
        file << velocity_scale

        # Max norm is perhaps better suited?
        if abs(new_norm - prev_norm) / new_norm < 100:
            dump_stats(eval_dict["senterline_u"].number_of_evaluations(), 
                       eval_dict, newfolder)

            if MPI.rank(mpi_comm_world()) == 0:
                kill = open(folder + '/killoasis', 'w')
                kill.close()

    if tstep % checkpoint == 0:
        dump_stats(eval_dict["senterline_u"].number_of_evaluations(),
                   eval_dict, newfolder)


def dump_stats(num_eval, eval_dict, newfolder):
    filepath = path.join(newfolder, "Stats")
    
    # Remove previous stats files
    if listdir(filepath) != []:
        if MPI.rank(mpi_comm_world()) == 0:
            for file in listdir(filepath):
                remove_path = path.join(filepath, file)
                remove(remove_path)
    MPI.barrier(mpi_comm_world())

    # Dump stats, store number of evaluations in filename
    for key, value in eval_dict.iteritems():
        if "slices" in key:
            for val in value:
                #z_point = str(val.get_probe(1).coordinates()[-1])
                val[0].array(filename=path.join(filepath, key + "_" \
                                       + str(val[-1]) + "_" + str(num_eval)))
        elif key != "initial_u":
            value.array(filename=path.join(filepath, key + "_" + str(num_eval)))


def print_header(dt, hmin, hmax, Re, start, stopp, inlet_velocity,
                 inlet_string, num_cell, folder, mesh_path):
    file = open(path.join(folder, "problem_paramters.txt"), "w")
    file.write("Writing header")
    file.write("=== Nozzle with sudden expanssioni ===")
    file.write("dt=%e" % dt)
    file.write("hmin=%e" % hmin)
    file.write("hmax=%s" % hmax)
    file.write("Re=%d" % Re)
    file.write("Start=%s" % start)
    file.write("Stopp=%s" % stopp)
    file.write("Inlet=%s" % inlet_string)
    file.write("u_0=%s" % inlet_velocity)
    file.write("Number of cells=%s" % num_cell)
    file.write("Path to mesh=%s" % mesh_path)
    file.close()
