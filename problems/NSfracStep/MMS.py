from ..NSfracStep import *
import numpy as np
import sympy as sp
import re
import sys
from sympy.printing.ccode import CCodePrinter
import time


############
# Taken from https://github.com/MiroK/fenics-ls/blob/master/sympy_utils.py
def is_sympy(args):
    'Check if args are sympy objects.'
    try:
        return args.__module__.split('.')[0] == 'sympy'
    except AttributeError:
        return False


class DolfinCodePrinter(CCodePrinter):
    '''
    This class provides functionality for converting sympy expression to
    DOLFIN expression. Core of work is done by dolfincode.
    '''
    def __init__(self, settings={}):
        CCodePrinter.__init__(self)

    def _print_Pi(self, expr):
        return 'pi'

def dolfincode(expr, assign_to=None, **settings):
    # Handle scalars
    if is_sympy(expr):
        dolfin_xs = sp.symbols('x[0] x[1] x[2]')
        xs = sp.symbols('x y z')
        for x, dolfin_x in zip(xs, dolfin_xs):
            expr = expr.subs(x, dolfin_x)
        return DolfinCodePrinter(settings).doprint(expr, assign_to)

    # Recurse if vector or tensor
    elif type(expr) is tuple:
        return tuple(dolfincode(e, assign_to, **settings) for e in expr)
#############


def find_f():
    x, y, z, t, nu, eps = sp.symbols('x y z t nu eps')
    #t_ = 1e-5
    #t_ = 1
    #u = 0.3*sp.tanh(sp.pi * (y+x+z)) + 0.4*sp.tanh(sp.pi*(3*y+2.3*z)) 
    #u = (2.1*sp.tanh(sp.pi*(x+y)) - 0.5*sp.tanh(sp.pi*y) + eps) * (t + 0.2)  
    u =  (sp.sin(sp.pi*(x+y+z)) + sp.cos(sp.pi*y*z))*t + eps
    #v = 0.3*sp.tanh(sp.pi*(y+x+z)) + 1.3*sp.tanh(sp.pi*(1.2*x+0.3*z))
    #v = (-2.1*sp.tanh(sp.pi*(x+y)) + 1.3*sp.tanh(sp.pi*x) + eps) * (t + 0.2)
    v = (-sp.sin(sp.pi*(x+y))  + sp.cos(sp.pi*x*z))*t + eps
    #p = (sp.tanh(sp.pi*(2.1*x+1.2*y+0.3*z)) + eps)  #*sp.sin(sp.pi*t) + eps    
    p = (eps * sp.sin(sp.pi*(x + y + z)) * sp.cos((x + y + z)*sp.pi) + eps) * (t + 0.3)
    #p = (eps * sp.tanh(sp.pi*(x + y)) + eps) 

    # An exact solution
    #u = 2*(x**2 + y**2 + z**2) + 0.5
    #v = 2.5*(x**2 + y**2 + z**2) + 1.
    #p = x + y + z + 0.5

    dudx = sp.diff(u,x)
    dvdy = sp.diff(v,y)
    
    w = -sp.integrate(dudx + dvdy, z) + eps # add something dependent on x and y

    var = [u, v, w]      # add w if 3D is wanted

    # Check if the velocity is divergence free
    if len(var) > 2: 
        print dudx + dvdy + sp.diff(w, z)
    else:
        print dudx + dvdy
    
    fx = sp.diff(u, t) + u*sp.diff(u, x) + v*sp.diff(u, y) + sp.diff(p, x) \
         - nu*(sp.diff(u, x, x) + sp.diff(u, y, y))
    fy = sp.diff(v, t) + u*sp.diff(v, x) + v*sp.diff(v, y) + sp.diff(p, y) \
         - nu*(sp.diff(v, x, x) + sp.diff(v, y, y))
    if len(var) > 2:
        fz = sp.diff(w, t) + u*sp.diff(w, x) + v*sp.diff(w, y) + w*sp.diff(w, z) \
             + sp.diff(p, z) - nu*(sp.diff(w, x, x) + sp.diff(w, y, y) + sp.diff(w, z, z))
        fx += w*sp.diff(u, z) - nu*sp.diff(u, z, z)
        fy += w*sp.diff(v, z) - nu*sp.diff(v, z, z)


    # Convert into cpp syntax
    f = (fx, fy) if len(var) == 2 else (fx, fy, fz)
    f = dolfincode(f)
    
    # Constants
    eps = 0.005
    nu = 0.1
    t = 0
    
    # Expression for the source term in the MMS
    f = Expression(f, nu=nu, eps=eps, t=t)
    
    # Expression for the velocity components
    u_ = []
    for var_ in var:
        u_.append(dolfincode(var_))
    p = dolfincode(p)

    exact_u = Expression(tuple(u_), eps=eps, nu=nu, t=t)
    exact_p = Expression(p, eps=eps, nu=nu, t=t)

    for i in range(len(u_)):
        u_[i] = Expression(u_[i], eps=eps, nu=nu, t=t)
    
    return exact_u, exact_p, u_,  f


def update(**NS_namespace):
    rhs = find_f()
    d = recursive_update(NS_parameters,
                    dict(nu=0.1,
                        T=1e10,
                        dt=0.1,
                        N=10,
                        folder="MMS_results",
                        save_tstep=1000e9,
                        checkpoint=1000e9,
                        check_steady=5,
                        velocity_degree=3,
                        pressure_degree=2,
                        u_e=rhs[0],
                        p_e=rhs[1],
                        u_seg=rhs[2],
                        source_term=rhs[3],
                        print_intermediate_info=1000,
                        use_lumping_of_mass_matrix=False,
                        max_iter=100,
                        plot_interval=1000,
                        max_error=1e-12,
                        low_memory_version=False,
                        use_krylov_solvers=True,
                        krylov_report=False,
                        krylov_solvers=dict(monitor_convergence=False,
                                        relative_tolerance=1e-10,
                                        absolute_tolerance=1e-10)))
    globals().update(d)


def mesh(u_seg, N, **NS_namespace):
    if len(u_seg) == 2:
        return UnitSquareMesh(N, N)
    else:
        return UnitCubeMesh(N, N, N)


def boundary(x, on):
    return on #and not right(x, on)


def right(x, on):
    return on #and near(x[0], 0)


def create_bcs(u_seg, p_e, sys_comp, V, Q, **NS_namespce):
    bcs = dict((ui, []) for ui in sys_comp)
    for i in range(len(u_seg)):
        bcs["u%s" % i] = [DirichletBC(V, u_seg[i], boundary)]
    bcs["p"] = [DirichletBC(Q, p_e, right)]

    return bcs


def initialize(q_1, q_, q_2, u_e, p_e, VV, u_seg, t, dt, **NS_namespace):
    #pass
    init = {"u0": u_seg[0], "u1": u_seg[1], "p": p_e}
    if len(u_seg) > 2:
        init["u2"] = u_seg[2]

    for ui in q_:
        init[ui].t = t + dt/2. if ui == "p" else t
        vv = interpolate(init[ui], VV[ui])
        q_[ui].vector()[:] = vv.vector()[:]
        if not ui == "p":
            q_1[ui].vector()[:] = vv.vector()[:]
            init[ui].t = t-dt
            vv = interpolate(init[ui], VV[ui])
            q_2[ui].vector()[:] = vv.vector()[:]
        q_1['p'].vector()[:] = q_['p'].vector()[:]


def body_force(source_term, **NS_namespace):
    return source_term


def pre_solve_hook(velocity_degree, mesh, dt, pressure_degree, u_e,
                   newfolder, p_e, N, **NS_namespace):
    
    Vv = VectorFunctionSpace(mesh, "CG", velocity_degree)
    Pv = FunctionSpace(mesh, "CG", pressure_degree)

    V5 = FunctionSpace(mesh, "CG", 5 + velocity_degree)

    error_u = {'u0': [1e10], 'u1': [1e10], 'u2': [1e10], 'p': [1e10]}

    return dict(error_u=error_u, V5=V5) 


def start_timestep_hook(t, u_seg, f, dt, p_e, u_e, **NS_namespace):
    for i in range(len(u_seg)):
        u_seg[i].t = t
    p_e.t = t - dt/2.
    u_e.t = t
    f.t = t - dt/2.


def temporal_hook(u_, p_, u_seg, p_e, check_steady, tstep, dt, t,
                  error_u, mesh, q_, folder, V5, sys_comp, **NS_namespace):

    print tstep
    if tstep % check_steady == 0 and tstep > 2:
        for i, ui in enumerate(sys_comp): 
            u_e = p_e if i == len(u_seg) else u_seg[i]
            u_e = interpolate(u_e, V5)
            uen = norm(u_e)
            diff = project(u_e - q_[ui], V5)
            error = norm(diff) / uen
            error_u[ui].append(error)

            print "%s: %s" % (ui, error_u[ui][-1])

            if tstep*dt >= 4:
               #abs(error_u['u0'][-1] - error_u['u0'][-2]) < 1e-9 and \
               #abs(error_u['u1'][-1] - error_u['u1'][-2]) < 1e-9 and \
               #abs(error_u['p'][-1] - error_u['p'][-2]) < 1e-9:
                kill = open(folder + '/killoasis', 'w')
                kill.close()


def theend_hook(error_u, mesh, **NS_namespace):
    print "Error u l2 min:", error_u
    print "hmin:", mesh.hmin()
