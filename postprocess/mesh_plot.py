from dolfin import *
import sys
from os import path

mesh_path = path.join(path.dirname(__file__), "..", "mesh", sys.argv[1])
mesh = Mesh(mesh_path)

V = FunctionSpace(mesh, "DG", 0)
h = CellVolume(mesh)
h = (sqrt(2)/12. * h)**(1./3)
u = project(h, V)

file = File(path.join(path.dirname(__file__), "..", "nozzle_results", "Mesh",
            sys.argv[1].split(".")[0] + ".pvd"))
file << u
