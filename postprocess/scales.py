from dolfin import *
from numpy import *


def stress(ssv, u, v, mesh)
    epsilon = 0.5*(grad(u) + grad(u).T)
    f = 2*mu*sqrt(inner(epsilon, epsilon))
    x = assemble(inner(f, v)/h*dx(mesh))
    ssv.vector().set_local(x.array())
    ssv.vector().apply("insert")

    return ssv

#TODO: Need nu

def compute_scales(mesh, files, nu):
    DG = FunctionSpace(mesh, 'DG', 0)
    V = VectorFunctionSpace(mesh, "CG", 1)

    v = TestFunction(DG)
    ssv = Function(DG)
    l_pluss_tmp = Function(DG)
    t_pluss_tmp = Function(DG)
    l_pluss = Function(DG)
    t_pluss = Function(DG)
    length_scale = Function(DG)
    time_scale = Function(DG)
    velocity_scale = Function(DG)

    h = CellVolume(mesh)
    dl = project(12/math.sqrt(2) * h**(1./3), DG)

    l_pluss = {"max": 0, "min"; 1e9, "mean": 0}
    t_pluss = {"max": 0, "min"; 1e9, "mean": 0}

    for file in files:
        u = Function(V, file)
        ssv = stress(ssv, u, v, mesh)
        
        u_star = ssv.vector().array() / (2 * rho)

        #TODO: Find max, min and mean of both l_pluss and Kolmogorov scales.
        #      Also need to have one function that is the mean of all meshes.

        # Compute l+ and t+
        l_pluss.vector().set_local(np.sqrt(u_star) * dl.vector().array() / nu)
        l_pluss.vector().apply("insert")

        t_pluss.vector().set_local(nu / u_star)
        t_pluss.vector().apply("insert")

        # Compute Kolmogorov
        length_scale.vector().set_local((nu**3 / ssv)**(1./4))
        length_scale.vector().apply("insert")

if __name__ == "__main__":
    #TODO: Read commandline
    #TODO: Find files
    #TODO: Find nu
    #TODO: Load mesh
    #TODO: compute_scales(mesh, files, nu)
