__author__ = 'Mikael Mortensen <mikaem@math.uio.no>'
__date__ = '2015-01-22'
__copyright__ = 'Copyright (C) 2015 ' + __author__
__license__ = 'GNU Lesser GPL version 3 or any later version'

import sys
from dolfin import MeshFunction, DirichletBC, Constant, MPI


def derived_bcs(V, original_bcs, u_):

    new_bcs = []

    # Check first if user has declared subdomains
    subdomain = original_bcs[0].user_subdomain()
    if subdomain is None:
        mesh = V.mesh()
        ff = MeshFunction("size_t", mesh, 0)
        for i, bc in enumerate(original_bcs):
            bc.apply(u_[0].vector())  # Need to initialize bc
            if not hasattr(bc, "markers"):
                if MPI.rank(MPI.comm_world) == 0:
                    print("ERROR: Python interface in FEniCS is missing DirichletBC.markers(), " \
                         + "can therefore not run correctly. Exit on 0.")
                sys.exit(0)
            m = bc.markers()  # Get facet indices of boundary
            ff.get_local()[m] = i + 1
            new_bcs.append(DirichletBC(V, Constant(0), ff, i + 1))

    else:
        for i, bc in enumerate(original_bcs):
            subdomain = bc.user_subdomain()
            new_bcs.append(DirichletBC(V, Constant(0), subdomain))

    return new_bcs
