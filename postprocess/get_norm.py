from re import *
from os import path
from matplotlib.pyplot import *
from argparse import ArgumentParser


def read_command_line():
    """Read arguments from commandline"""
    parser = ArgumentParser()  

    parser.add_argument('--f', type=str, default=".", 
                        help="Path to the log for the runs")

    args = parser.parse_args()

    return args.f


def get_norm(filename):
    f = open(filename, "r")
    text = f.read()
    f.close()

    # Split on each restart and assume that the entire log is from the same
    # simulation
    different_runs = text.split("Starting job")

    # First start include first norm
    norm = findall(r"Norm:\s(.*)\n", different_runs[0])

    print norm
    
    # For all other runs skip the first norm comparison as it is so large.
    print range(1, len(different_runs))
    for i in range(1, len(different_runs)):
        tmp_text = different_runs[i]
        norm += findall(r"Norm:\s(.*)$", tmp_text)[0][1:]

    t = linspace(0, len(norm)*300, 300)

    plot(t, norm)
    xlabel("Timesteps")
    ylabel("Residual norm")
    savefig(path.join(path.dirname(filename), "norm.png"))

if __name__ == "__main__":
    filename = read_command_line()
    get_norm(filename)
