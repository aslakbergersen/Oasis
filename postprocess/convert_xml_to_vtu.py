import sys
import re
from os import path, listdir
from dolfin import Mesh, Function, FunctionSpace, VectorFunctionSpace, File


def main(folder):
    basedir = path.dirname(path.abspath(__file__))
    folder_path = path.join(basedir, "..", "nozzle_results", "data", folder)
    mesh_path = read_problem_parameters(folder_path)
    convert_xml_to_vtu(mesh_path, folder_path)


def convert_xml_to_vtu(mesh, folder):
    mesh = path.join(path.dirname(path.abspath(__file__)), "..", mesh)
    mesh = Mesh(mesh)
    
    files = listdir(path.join(folder, "VTK"))
    last_timestep = max([int(f[-13:-7]) for f in files if ".xml.gz" in f])
    file_names = ["length_scale%06.0f.xml.gz", 
                  "pressure%06.0f.xml.gz",
                  "stress%06.0f.xml.gz",
                  "time_scale%06.0f.xml.gz",
                  "velocity%06.0f.xml.gz"]
    file_names = [path.join(folder, "VTK", f) % last_timestep for f in file_names]
    V_vec = VectorFunctionSpace(mesh, "CG", 1)
    V_seg = FunctionSpace(mesh, "CG", 1)
    V_seg_dg = FunctionSpace(mesh, "DG", 0)

    for name in file_names:
        file = File(name.replace(".xml.gz", ".pvd"))

        try:
            if "velocity" in name:
                comp = Function(V_vec, name)
            elif "_scale" in name or "stress" in name:
                comp = Function(V_seg_dg, name)
            else:
                comp = Function(V_seg, name)
        except:
            print name, "could not convert"

        comp.rename(name[:-13], "")
        file << comp


def read_problem_parameters(folder):
    f = open(path.join(folder, "problem_parameters.txt"), "r")
    t = f.read()
    f.close()
    mesh_path = re.findall(r"Path to mesh=(.*)", t)[0]
    return mesh_path


if __name__ == "__main__":
    if len(sys.argv) == 2:
        folder = sys.argv[1]
        main(folder)
    else:
        print "Usage %s [folder]" % sys.argv[0]
        print sys.exit(0)

