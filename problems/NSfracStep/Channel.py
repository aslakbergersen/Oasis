__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from ..NSfracStep import *
from fenicstools import StructuredGrid, Probes
from numpy import arctan, array, cos, pi
from os import getcwd
import cPickle
import random

restart_folder = 'channel_results/data/2/Checkpoint'
restart_folder = None

class ChannelGrid(StructuredGrid):
    """Grid for computing statistics"""
    def modify_mesh(self, dx, dy, dz):
        """Create grid skewed towards the walls located at y = 1 and y = -1"""
        dy[1][:] = arctan(pi*(dy[1][:]+self.origin[1])) / arctan(pi) - self.origin[1]
        return dx, dy, dz

def mesh(Nx, Ny, Nz, Lx, Ly, Lz, **params):
    # Function for creating stretched mesh in y-direction
    m = BoxMesh(0., -Ly/2., -Lz/2., Lx, Ly/2., Lz/2., Nx, Ny, Nz)
    x = m.coordinates() 
    x[:, 1] = arctan(1.*pi*(x[:, 1])) / arctan(1.*pi) 
    return m

### If restarting from previous solution then read in parameters ########
if restart_folder:
    restart_folder = path.join(getcwd(), restart_folder)
    f = open(path.join(restart_folder, 'params.dat'), 'r')
    NS_parameters.update(cPickle.load(f))
    NS_parameters['T'] = NS_parameters['T'] + 200 * NS_parameters['dt'] 
    NS_parameters['restart_folder'] = restart_folder
    globals().update(NS_parameters)
    
else:
    Lx = 4.*pi
    Ly = 2.
    Lz = 4.*pi/3.
    Nx = 16
    Ny = 16
    Nz = 16
    NS_parameters.update(Lx=Lx, Ly=Ly, Lz=Lz, Nx=Nx, Ny=Ny, Nz=Nz)
    
    # Override some problem specific parameters
    T = 100.
    dt = 0.2
    nu = 2.e-5
    Re_tau = 178.12
    NS_parameters.update(
        update_statistics = 10,
        save_statistics = 100,
        check_flux = 10,
        checkpoint = 10,
        save_step = 100,
        nu = nu,
        Re_tau = Re_tau,
        T = T,
        dt = dt,
        velocity_degree = 1,
        folder = "channel_results",
        use_krylov_solvers = True
    )

##############################################################
class PeriodicDomain(SubDomain):

    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary AND NOT on one of the two slave edges
        return bool((near(x[0], 0) or near(x[2], -Lz/2.)) and 
                (not (near(x[0], Lx) or near(x[2], Lz/2.))) and on_boundary)
                      
    def map(self, x, y):
        if near(x[0], Lx) and near(x[2], Lz/2.):
            y[0] = x[0] - Lx
            y[1] = x[1] 
            y[2] = x[2] - Lz
        elif near(x[0], Lx):
            y[0] = x[0] - Lx
            y[1] = x[1]
            y[2] = x[2]
        else: # near(x[2], Lz/2.):
            y[0] = x[0]
            y[1] = x[1]
            y[2] = x[2] - Lz
            
constrained_domain = PeriodicDomain()

def inlet(x, on_bnd):
    return on_bnd and near(x[0], 0)

# Specify body force
utau = nu * Re_tau
def body_force(**NS_namespace):
    return Constant((utau**2, 0., 0.))

def pre_solve_hook(V, q_, q_1, q_2, u_components, mesh, **NS_namespace):    
    """Called prior to time loop"""
    Vv = VectorFunctionSpace(V.mesh(), 'CG', 1, constrained_domain=constrained_domain)
    uv = Function(Vv) 
    tol = 5e-8
    
    # It's periodic so don't pick the same location twice for sampling statistics:
    stats = ChannelGrid(V, [Nx, Ny+1, Nz], [tol, -Ly/2., -Lz/2.+tol], [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], [Lx-Lx/Nx, Ly, Lz-Lz/Nz], statistics=True)
    
    # Create FacetFunction to compute flux
    Inlet = AutoSubDomain(inlet)
    facets = FacetFunction('size_t', mesh, 0)
    Inlet.mark(facets, 1)    
    normal = FacetNormal(mesh)

    return dict(uv=uv, stats=stats, facets=facets, normal=normal)
    
def walls(x, on_bnd):
    return (near(x[1], -Ly/2.) or near(x[1], Ly/2.))

def create_bcs(V, q_, q_1, q_2, sys_comp, u_components, **NS_namespace):
    info_red("Creating boundary conditions")
    bcs = dict((ui, []) for ui in sys_comp)    
    bc = [DirichletBC(V, Constant(0), walls)]
    bcs['u0'] = bc
    bcs['u1'] = bc
    bcs['u2'] = bc
    return bcs

class RandomStreamVector(Expression):
    def __init__(self):
        random.seed(2 + MPI.rank(mpi_comm_world()))
    def eval(self, values, x):
        values[0] = 0.0005*random.random()
        values[1] = 0.0005*random.random()
        values[2] = 0.0005*random.random()
    def value_shape(self):
        return (3,)  

def initialize(V, q_, q_1, q_2, bcs, restart_folder, **NS_namespace):
    if restart_folder is None:
        # Initialize using a perturbed flow. Create random streamfunction
        Vv = VectorFunctionSpace(V.mesh(), V.ufl_element().family(), V.ufl_element().degree())
        psi = interpolate(RandomStreamVector(), Vv)
        u0 = project(curl(psi), Vv)
        u0x = project(u0[0], V, bcs=bcs['u0'])
        u1x = project(u0[1], V, bcs=bcs['u0'])
        u2x = project(u0[2], V, bcs=bcs['u0'])
        
        # Create base flow
        y = interpolate(Expression("x[1] > 0 ? 1-x[1] : 1+x[1]"), V)
        uu = project(1.25*(utau/0.41*ln(conditional(y<1e-12, 1.e-12, y)*utau/nu)+5.*utau), V, bcs=bcs['u0'])
        
        # initialize vectors at two timesteps
        q_1['u0'].vector()[:] = uu.vector()[:] 
        q_1['u0'].vector().axpy(1.0, u0x.vector())
        q_1['u1'].vector()[:] = u1x.vector()[:]
        q_1['u2'].vector()[:] = u2x.vector()[:]
        q_2['u0'].vector()[:] = q_1['u0'].vector()[:]
        q_2['u1'].vector()[:] = q_1['u1'].vector()[:]
        q_2['u2'].vector()[:] = q_1['u2'].vector()[:]
    
def tentative_velocity_hook(ui, use_krylov_solvers, u_sol, **NS_namespace):
    if use_krylov_solvers:
        # Make tolerance stricter in x-direction (direction of flow)
        if ui == "u0":            
            u_sol.parameters['preconditioner']['structure'] = "same_nonzero_pattern"
        else:
            u_sol.parameters['preconditioner']['structure'] = "same"

def temporal_hook(q_, u_, V, tstep, uv, stats, update_statistics,
                  newfolder, folder, check_flux, save_statistics, mesh,
                  facets, normal, check_if_reset_statistics, **NS_namespace):
    # print timestep
    info_red("tstep = {}".format(tstep))         
    if check_if_reset_statistics(folder):
        info_red("Resetting statistics")
        stats.probes.clear()

    if tstep % update_statistics == 0:
        stats(q_['u0'], q_['u1'], q_['u2'])

    if tstep % save_statistics == 0:
        statsfolder = path.join(newfolder, "Stats")
        stats.toh5(0, tstep, filename=statsfolder+"/dump_mean_{}.h5".format(tstep))
        
    if tstep % check_flux == 0:
        u1 = assemble(dot(u_, normal)*ds(1, domain=mesh, subdomain_data=facets))
        u1 = assemble(dot(u_, normal)*ds(1, domain=mesh, subdomain_data=facets))
        normv = norm(q_['u1'].vector())
        normw = norm(q_['u2'].vector())
        if MPI.rank(mpi_comm_world()) == 0:
           print "Flux = ", u1, " tstep = ", tstep, " norm = ", normv, normw
        
def theend(newfolder, tstep, stats, **NS_namespace):
    """Store statistics before exiting"""
    statsfolder = path.join(newfolder, "Stats")
    stats.toh5(0, tstep, filename=statsfolder+"/dump_mean_{}.h5".format(tstep))
