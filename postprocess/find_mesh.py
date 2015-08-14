from dolfin import *
from argparse import ArgumentParser
from os import path, listdir
import numpy as np
import sys


def read_command_line():
    """Read arguments from commandline"""
    parser = ArgumentParser()

    parser.add_argument('--M', type=str, default="28",
                        help="Assume that the mesh is calles XM, where X is the " + \
                             "argument to provide here")
    parser.add_argument('--P', type=str, default=1, help="Velocity degree")
    parser.add_argument('--run', type=str, default="4", help="The ID number of the run, " + \
                        "e.g. data/ID/")

    args = parser.parse_args()

    return args.M, args.P, args.run


def get_stats(f):
    vec = f.vector().array()
    mean = np.mean(vec)
    median = np.median(vec)
    max = np.max(vec)   
    min = np.min(vec)
    return min, max, median, mean


def main(M, P, run):
    # Independent of working dir
    file_path = path.dirname(path.abspath(__file__))

    # Load mesh
    mesh = Mesh(path.join(file_path, "..", "mesh", "%sM.xml" % M))
    
    # Function spaces
    V = VectorFunctionSpace(mesh, "CG", 1)
    DG = FunctionSpace(mesh, "DG", 0)
    
    # Look at dx
    h = project(CellSize(mesh), DG)
    dx_stats = get_stats(h)

    # Look at l+ and t+
    VTK_path = path.join(file_path, "..", "VTK")
    plots = listdir(VTK_path)
    
    # Find latest l+ plot
    l_pluss = [path.join(VTK_path, p) for p in plots if "length_scale" in p]
    l_pluss.sort()
    l_pluss_path = l_pluss[0]

    # Look at l+
    l_pluss = Function(DG, l_pluss_path)
    l_pluss_stats = get_stats(l_pluss)

    # Find latest t+ plot
    t_pluss = [path.join(VTK_path, p) for p in plots if "time_scale" in p]
    t_pluss.sort()
    t_pluss_path = t_pluss[0]

    # Look at t+ plot
    t_pluss = Function(DG, t_pluss_path)
    t_pluss_stats = get_stats(t_pluss)

    legend = ["dx", "time scale", "length scale"]
    values = [dx_stats, t_pluss_stats, l_pluss_stats]
    for i in range(3):
        if MPI.rank(mpi_comm_world()) == 0:
            print "-"*10, legend[i], "-"*10
            print "Min       %e" % values[i][0]
            print "Max       %e" % values[i][1]
            print "Median    %e" % values[i][2]
            print "Mean      %e" % values[i][3]
            print ""


if __name__ == "__main__":
    M, P, run = read_command_line()
    main(M, P, run)
