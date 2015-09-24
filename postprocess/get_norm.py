from re import *
from os import path, listdir
from matplotlib.pyplot import *
from argparse import ArgumentParser
import numpy as np


def read_command_line():
    """Read arguments from commandline"""
    parser = ArgumentParser()  

    parser.add_argument('--f', type=str, default=None, 
                        help="Name of the run, e.g. 19")

    args = parser.parse_args()

    return args.f


def get_norm(folder):
    """Collect data from the computations"""
    folder_path = path.join(path.dirname(__file__), "..", "nozzle_results", "data")

    # Find latest run
    if folder is None:
        folders = listdir(folder_path)
        folder = array([int(f) for f in folders]).max()
    
    logsfolder_path = path.join(folder_path, folder, "logs")
    logs_path = [path.join(logsfolder_path, f) for f in listdir(logsfolder_path)]
    s = ""
    for log in logs_path:
        f = open(log, "r")
        s += f.read()
        f.close()

    # Get all norms and make "timeline"
    norm = findall(r"Norm:\s(.*)\n", s)
    norm = [n for n in norm[3:] if float(n) < 1.]
    t = np.linspace(0, len(norm)*300, len(norm))

    print(len(norm))
    print t.shape

    plot(t, norm)
    xlabel("Timesteps")
    ylabel("Residual norm")
    savefig(path.join(logsfolder_path, "norm.png"))

if __name__ == "__main__":
    folder = read_command_line()
    get_norm(folder)
