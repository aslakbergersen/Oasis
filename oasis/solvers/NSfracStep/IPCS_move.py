__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-11-07"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"
"""This is a simplest possible naive implementation of the IPCS solver.

The idea is that this solver can be quickly modified and tested for
alternative implementations. In the end it can be used to validate
the implementations of the more complex optimized solvers.

"""
from dolfin import *
from ..NSfracStep import *
from ..NSfracStep import __all__

def setup(u, q_, q_1, uc_comp, u_components, dt, v, U_AB, u_1, u_2, q_2,
          nu, p_, wu_, mesh, f, fs, q, p, u_, Schmidt, Schmidt_T, les_model, nut_,
          scalar_components, dp_, **NS_namespace):
    """Set up all equations to be solved."""
    # Implicit Crank Nicolson velocity at t - dt/2
    U_CN = dict((ui, 0.5 * (u + q_1[ui])) for ui in uc_comp)

    F = {}
    Fu = {}
    for i, ui in enumerate(u_components):
        # Tentative velocity step
        if not les_model is "NoModel":
            F[ui] = ((1. / dt) * inner(u - q_1[ui], v) * dx
                     + inner(dot(U_AB - wu_, nabla_grad(U_CN[ui])), v) * dx
                     + (nu + nut_) * inner(grad(U_CN[ui]), grad(v)) * dx
                     + inner(p_.dx(i), v) * dx - inner(f[i], v) * dx
                     + (nu + nut_) * inner(grad(v), U_AB.dx(i)) * dx)
        else:
            F[ui] = ((1. / dt) * inner(u - q_1[ui], v) * dx
                     + inner(dot(U_AB - wu_, nabla_grad(U_CN[ui])), v) * dx
                     + nu * inner(grad(U_CN[ui]), grad(v)) * dx
                     + inner(p_.dx(i), v) * dx - inner(f[i], v) * dx)

        # Velocity update
        Fu[ui] = (inner(u, v) * dx - inner(q_[ui], v) * dx
                  + dt * inner(dp_.dx(i), v) * dx)

    # Pressure update
    Fp = (inner(grad(q), grad(p)) * dx - inner(grad(p_), grad(q)) * dx
          + (1. / dt) * div(u_) * q * dx)

    # Scalar with SUPG
    h = CellDiameter(mesh)
    #vw = v + h*inner(grad(v), U_AB)
    vw = v
    n = FacetNormal(mesh)
    for ci in scalar_components:
        F[ci] = ((1. / dt) * inner(u - q_1[ci], vw) * dx
                 + inner(dot(grad(U_CN[ci]), U_AB), vw)*dx
                 + (nu / Schmidt[ci] + nut_ / Schmidt_T[ci]) * inner(grad(U_CN[ci]), grad(vw)) * dx
                 - inner(fs[ci], vw) * dx)

    au, Lu = system(F["u0"])
    Au = assemble(au)
    bu = assemble(Lu)

    auu, Luu = lhs(Fu["u0"]), rhs(Fu["u0"])
    Auu = assemble(auu)
    buu = assemble(Luu)

    ap, Lp = lhs(Fp), rhs(Fp)
    Ap = assemble(ap)
    bp = assemble(Lp)
    return dict(F=F, Fu=Fu, Fp=Fp, Au=Au, bu=bu, Ap=Ap, bp=bp, Auu=Auu, buu=buu)


def get_solvers(use_krylov_solvers, krylov_solvers, bcs,
                x_, Q, scalar_components, velocity_krylov_solver,
                pressure_krylov_solver, scalar_krylov_solver, **NS_namespace):
    """Return linear solvers.

    We are solving for
       - tentative velocity
       - pressure correction

       and possibly:
       - scalars

    """
    if use_krylov_solvers:
        ## tentative velocity solver ##
        u_prec = PETScPreconditioner(velocity_krylov_solver['preconditioner_type'])
        u_sol = PETScKrylovSolver(velocity_krylov_solver['solver_type'], u_prec)
        u_sol.parameters.update(krylov_solvers)

        p_sol = KrylovSolver(pressure_krylov_solver['solver_type'],
                             pressure_krylov_solver['preconditioner_type'])
        #p_sol.parameters['preconditioner']['structure'] = 'same'
        #p_sol.parameters['profile'] = True
        p_sol.parameters.update(krylov_solvers)

        sols = [u_sol, p_sol]

        sols.append(None)
    else:
        ## tentative velocity solver ##
        u_sol = LUSolver()
        u_sol.parameters['same_nonzero_pattern'] = True
        ## pressure solver ##
        p_sol = LUSolver()
        p_sol.parameters['reuse_factorization'] = True
        if bcs['p'] == []:
            p_sol.normalize = True
        sols = [u_sol, p_sol]
        ## scalar solver ##
        if len(scalar_components) > 0:
            c_sol = LUSolver()
            sols.append(c_sol)
        else:
            sols.append(None)

    return sols


def velocity_tentative_solve(ui, F, q_, bcs, Au, bu, x_, b_tmp, udiff, u_sol, **NS_namespace):
    """Linear algebra solve of tentative velocity component."""
    b_tmp[ui][:] = x_[ui]
    a, L = system(F[ui])

    t1 = Timer("Assemble tentative velocity")
    assemble(a, tensor=Au)
    assemble(L, tensor=bu)
    t1.stop()

    [(bc.apply(bu), bc.apply(Au)) for bc in bcs[ui]]

    t1 = Timer("Tentative linear algebra solve")
    u_sol.solve(Au, x_[ui], bu)
    t1.stop()
    #solve(A == L, q_[ui], bcs[ui])

    udiff[0] += norm(b_tmp[ui] - x_[ui])


def pressure_solve(Fp, p_, bcs, dp_, Ap, bp, x_, u_, q_, Q, p_sol, **NS_namespace):
    """Solve pressure equation."""
    dp_.vector()[:] = x_['p']

    t1 = Timer("Assemble pressure solve")
    assemble(lhs(Fp), tensor=Ap)
    assemble(lhs(Fp), tensor=bp)
    t1.stop()

    if bcs['p'] == []:
        normalize(p_.vector())
    else:
        [(bc.apply(Ap), bc.apply(bp)) for bc in bcs["p"]]

    p_sol.solve(Ap, x_["p"], bp)
    dp_.vector()[:] = - dp_.vector()[:]
    dp_.vector().axpy(1.0, x_['p'])


def velocity_update(u_components, q_, x_, bcs, Fu, Auu, buu, dp_, V, dt, u_sol, **NS_namespace):
    """Update the velocity after finishing pressure velocity iterations."""
    for ui in u_components:
        #t1 = Timer("Velocity assemble " + ui)
        #assemble(lhs(Fu[ui]), tensor=Auu)
        #assemble(rhs(Fu[ui]), tensor=buu)
        #t1.stop()

        #[(bc.apply(buu), bc.apply(Auu)) for bc in bcs[ui]]

        #t1 = Timer("Velocity solve" + ui)
        #u_sol.solve(Auu, x_[ui], buu)
        #t1.stop()
        solve(lhs(Fu[ui]) == rhs(Fu[ui]), q_[ui], bcs[ui])

def scalar_solve(ci, F, q_, bcs, **NS_namespace):
    """Solve scalar equation."""
    solve(lhs(F[ci]) == rhs(F[ci]), q_[ci], bcs[ci])
