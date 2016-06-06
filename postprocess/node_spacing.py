from dolfin import *
import sys
from os import path

def meshsize(mesh_path):
    print mesh_path
    mesh = Mesh(mesh_path)
    DG = FunctionSpace(mesh, "DG", 0)

    h = CellVolume(mesh)
    dl = project((12/math.sqrt(2) * h)**(1./3), DG)
    
    file = File(mesh_path.split(".")[0] + "_nodespacing.pvd")
    file << dl
    del file


if __name__ == "__main__":
    dirpath = path.join(path.dirname(path.abspath(__file__)), "..", "mesh")
    for s in [5, 10, 17, 28]:
        meshsize(path.join(dirpath, "%sM.xml" % s))
