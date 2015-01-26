from ..NSfracStep import *
from math import pi
from os import path, getcwd, listdir, remove
from numpy import array, linspace
import sys
import numpy as np
import cPickle
from mpi4py.MPI import COMM_WORLD as comm

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
inlet_string = 'u_0' # * (1 - (x[0]*x[0] + x[1]*x[1])/(r_0*r_0))'
restart_folder = None #"nozzle_results/data/3/Checkpoint"

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
                        dt=5E-6,
                        folder="nozzle_results",
                        case=3500,
                        save_tstep=1000,
                        checkpoint=1000,
                        check_steady=1,
                        eval_t=50,
                        velocity_degree=1,
                        pressure_degree=1,
                        mesh_path="mesh/1600K_opt_nozzle.xml",
                        print_intermediate_info=1000,
                        use_lumping_of_mass_matrix=False,
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


def create_bcs(V, sys_comp, nu, case, **NS_namespce):
    u_0 = nu * case / (2*r_0)   # Need to find mesh inlet area
    print u_0
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


def pre_solve_hook(velocity_degree, mesh, dt, pressure_degree, V,
		   mu, case, newfolder, mesh_path, **NS_namesepace):

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

    # Container for all evaluations points
    eval_dict = {}
    key_u = "slice_u_%s"
    key_ss = "slice_ss_%s"

    eps = 1e-8
    for i in range(len(z)):
        # Set up dict for the slices
        u_ = key_u % z[i]
        ss_ = key_ss % z[i]
        eval_dict[u_]= {'points':0, 'array': zeros((200,3)), 'num': 0}
        eval_dict[ss_] = {'points':0, 'array': zeros(200), 'num': 0}

        # Create eval points
        slices_points = linspace(-radius[i]+eps, radius[i]-eps, 200)
        points = array([[x, 0, z[i]] for x in slices_points])
        points.dump(path.join(newfolder, "Stats", "Points", "slice_%s" % z[i]))
	eval_dict[u_]["points"] = points
        eval_dict[ss_]["points"] = points

    # Setup probes in the centerline and at the wall
    z_senterline = linspace(start+eps, stop-eps, 10000)
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

    # Create probes on senterline and wall
    eval_dict["senterline_u"] = {"points":eval_senter, 
                                 "array": zeros((10000,3)),
                                 "num": 0,
                                 "array_prev": 0,
                                 "num_prev": 1}
    eval_dict["senterline_p"] = {"points":eval_senter,
                                 "array": zeros(10000),
                                 "num": 0}
    eval_dict["senterline_ss"] = {"points":eval_senter,
                                  "array": zeros(10000),
                                  "num": 0}
    eval_dict["initial_u"] = {"points":eval_senter,
                                 "array": zeros((10000,3)),
                                 "num": 0,
                                 "array_prev": 0,
                                 "num_prev": 1}
    eval_dict["wall_p"] = {"points":eval_wall,
                           "array": zeros(10000),
                           "num": 0}
    eval_dict["wall_ss"] = {"points":eval_wall,
                            "array": zeros(10000),
                            "num": 0}

    if restart_folder is None:
        # Print header
        if MPI.rank(mpi_comm_world()) == 0:
            u_0 = 2*flow_rate[case] / (r_0*r_0*pi)
            print_header(dt, mesh.hmax(), mesh.hmin(), case, start, stop, u_0,
                         inlet_string, mesh.num_cells(), newfolder, mesh_path)

    else:
        # Restart stats
        files = listdir(path.join(newfolder, "Stats"))
        eval = files[0].split("_")[-1]
        if files != []:
            for file in files:
                file_split = file.split("_")
                key = "_".join(file_split[:-1])
                arr = np.load(path.join(newfolder, "Stats", file))
                eval_dict[key]["array"] = arr
                eval_dict[key]["num"] = eval
                if key == "initial_u_%s" % eval:
                    eval_dict.pop("initial_u")
        else:
            if MPI.rank(mpi_comm_world()) == 0:
                print "WARNING:  The stats folder is empty and the stats is not restarted"

    def stress(u):
        def epsilon(u):
            return 0.5*(grad(u) + grad(u).T)
        return project(2*mu*sqrt(inner(epsilon(u),epsilon(u))), DG)
    
    def norm_l(u, l=2):
        if l == "max":
            return np.max(abs(u.flatten()))
        else:
            return np.sum(u**l)**(1./l)

    ########### ADD FLUX FOR SERIAL TEST ##################
    # LagrangeInterpolator for later use
    li = LagrangeInterpolator()

    # Box as a basis for a slice
    mesh = BoxMesh(-r_1, -r_1, -r_1, r_1, r_1, r_1, 100, 100, 100)
    bmesh = BoundaryMesh(mesh, "exterior")

    # Create SubMesh for side at z=0
    # This will be a UnitSquareMesh with topology dimension 2 in 3 space
    # dimensions
    cc = CellFunction('size_t', bmesh, 0)
    xyplane = AutoSubDomain(lambda x: x[2] < -r_1 + DOLFIN_EPS)
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

    ####################### END ###########################
    
    uv=Function(Vv)
    pv=Function(Pv)
    eval_map = {"p": pv, "ss": stress, "u": uv}
    eval_map["p"].rename("pressure", "pressure of nozzle")
    eval_map["u"].rename("velocity", "velocity of nozzle")

    file_u = File(path.join(newfolder, "VTK", "velocity.pvd"))
    file_p = File(path.join(newfolder, "VTK", "pressure.pvd"))
    file_ss = File(path.join(newfolder, "VTK", "stress.pvd"))
    file_l = File(path.join(newfolder, "VTK", "length_scale.pvd"))
    file_t = File(path.join(newfolder, "VTK", "time_scale.pvd"))
    file_v = File(path.join(newfolder, "VTK", "velocity_scale.pvd"))
    files = {"u": file_u, "p": file_p, "ss": file_ss,
             "t": file_t, "l": file_l, "v": file_v}

    return dict(Vv=Vv, Pv=Pv, eval_map=eval_map,DG=DG, z=z, files=files,
                norm_l=norm_l, eval_dict=eval_dict, flux=flux)
    
    
def temporal_hook(u_, p_, newfolder, mesh, check_steady, Vv, Pv, tstep, eval_dict, 
                norm_l, eval_map, dt, checkpoint, nu, z, mu, DG, eval_t,
                files, flux, **NS_namespace):

    if tstep % eval_t == 0 and eval_dict.has_key("initial_u"):
        evaluate_points(eval_dict, eval_map, u=u_)
        if MPI.rank(mpi_comm_world()) == 0:
            print tstep

    if tstep % check_steady == 0 and eval_dict.has_key("initial_u"): 
        # Store vtk files for post prosess in paraview 
        eval_map["u"].assign(project(u_, Vv))
        file = File(newfolder + "/VTK/nozzle_velocity_%06d.pvd" % (tstep))
        file << eval_map["u"]

        # Evaluate points
        if tstep % eval_t != 0:
            evaluate_points(eval_dict, eval_map, u=u_)

        print "Flux at each slice"
        for z_ in [start] + z + [stop]:
            print "%e %e" % (z_, flux(eval_map["u"], z=z_))
        
        # Check the max norm of the difference
        arr = eval_dict["initial_u"]["array"] / eval_dict["initial_u"]["num"] - \
               eval_dict["initial_u"]["array_prev"] / eval_dict["initial_u"]["num_prev"]
        norm = norm_l(arr, l="max")

        # Update prev 
        eval_dict["initial_u"]["array_prev"] = eval_dict["initial_u"]["array"].copy()
        eval_dict["initial_u"]["num_prev"] = eval_dict["initial_u"]["num"]

        # Print info
        if MPI.rank(mpi_comm_world()) == 0:
            print "Condition:", norm < 1,
            print "On timestep:", tstep,
            print "Norm:", norm

        # Initial conditions is "washed away"
        if norm < 1:
            if MPI.rank(mpi_comm_world()) == 0:
                print "="*25 + "\n DONE WITH FIRST ROUND\n" + "="*25
            eval_dict.pop("initial_u")
        
    if not eval_dict.has_key("initial_u"):
        # Project velocity, pressure and stress
        eval_map["u"].assign(project(u_, Vv))
        eval_map["p"].assign(project(p_, Pv))
        ssv = eval_map["ss"](eval_map["u"])
        ssv.rename("stress", "shear stress in nozzle")

        # Evaluate points
        evaluate_points(eval_dict, eval_map, u=ssv)

        # Check the max norm of the difference
        arr = eval_dict["senterline_u"]["array"] / eval_dict["senterline_u"]["num"] - \
                eval_dict["senterline_u"]["array_prev"] / eval_dict["senterline_u"]["num_prev"]
        #TODO: Compute the norm every time step?
        norm = norm_l(arr, l="max")

        # Update prev 
        eval_dict["senterline_u"]["array_prev"] = eval_dict["senterline_u"]["array"].copy()
        eval_dict["senterline_u"]["num_prev"] = eval_dict["senterline_u"]["num"]

        # TODO: Do this every timestep?
        # Compute scales for mesh evaluation
        nu = Constant(nu)
        mu = Constant(mu)

        epsilon = project(ssv / (2*mu) * sqrt(nu*2), DG)

        time_scale = project(sqrt(sqrt(nu) * epsilon), DG)
        length_scale = project(sqrt(sqrt(nu**3) / epsilon), DG)
        velocity_scale = project(sqrt(nu) / epsilon, DG)

        time_scale.rename("t", "time scale")
        length_scale.rename("l", "length scale")
        velocity_scale.rename("v", "velocity scale")

        # Store vtk files for post prosess in paraview 
        t_ = T * tstep
        files["u"] << eval_map["u"], t_
        files["l"] << length_scale, t_
        files["t"] << time_scale, t_
        files["v"] << velocity_scale, t_
        
        # Print info
        if MPI.rank(mpi_comm_world()) == 0:
            print "Condition:", norm < 1,
            print "On timestep:", tstep,
            print "Norm:", norm

        # Check if stats have stabilized
        if norm < 0.5:
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
        value["array"].dump(path.join(filepath, key + "_" + str(value["num"])))


def evaluate_points(eval_dict, eval_map, u=None):
    if eval_dict.has_key("initial_u"):
        for i in range(len(eval_dict["initial_u"]["points"])):
            x = eval_dict["initial_u"]["points"][i]
            try:
                rank = MPI.rank(mpi_comm_world())
                tmp = array([u[0](x), u[1](x), u[2](x)])
            except:
                tmp = 0
                rank = 0
            rank = MPI.max(mpi_comm_world(), rank)
            tmp = comm.bcast(tmp, root=rank)
            eval_dict["initial_u"]["array"][i] += tmp
        eval_dict["initial_u"]["num"] += 1

    else:
        for key, value in list(eval_dict.iteritems()):
            sample = eval_map[key.split("_")[1]]
            sample = sample if not type(sample) == type(lambda x: 1) else u
            for i in range(len(eval_dict[key])):
                try:
                    rank = MPI.rank(mpi_comm_world())
                    tmp = sample(value["points"][i])
                except:
                    tmp = 0
                    rank = 0
                rank = MPI.max(mpi_comm_world(), rank)
                tmp = comm.bcast(tmp, root=rank)
                eval_dict[key]["array"][i] += tmp

            eval_dict[key]["num"] += 1


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
