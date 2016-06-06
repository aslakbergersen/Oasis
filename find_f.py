from dolfin import *
import numpy as np
import sympy as sp
from sympy.printing.ccode import CCodePrinter

###### - ######
##  Taken from https://github.com/MiroK/fenics-ls/blob/master/sympy_utils.py
###### - ######
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
##### - #######


def find_f():
    x, y, z, t, nu, eps = sp.symbols('x y z t nu eps')
    u =  0.3*(sp.sin(sp.pi*(x+y+z)) + sp.cos(sp.pi*y*z))*t + eps
    v = 0.9*(-sp.cos(sp.pi*(x+y+z))  + sp.cos(sp.pi*x*z))*t + eps
    p = (eps * sp.sin(sp.pi*(x + y + z)) * sp.cos((x + y + z)*sp.pi) + eps) * (t + 0.3)

    # An exact solution
    #u = 2*(x**2 + y**2 + z**2) + 0.5
    #v = 2.5*(x**2 + y**2 + z**2) + 1.
    #p = x + y + z + 0.5

    dudx = sp.diff(u,x)
    dvdy = sp.diff(v,y)
    
    w = -sp.integrate(dudx + dvdy, z) + sp.cos(sp.pi*x*y) + eps # add something dependent on x and y
    print w
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
    print f
    print "w:", w
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

if __name__ == "__main__":
    u_e, p_e, u_, f = find_f()
    print f
