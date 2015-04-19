from argparse import ArgumentParser
from os import path, makedirs
from compare import *
from calculate_flux import *
#from validation_matrix import *
from fancy_plot import *

def read_command_line():
    """Read arguments from commandline"""
    parser = ArgumentParser()

    # Define arguments
    parser.add_argument('--c', '--compare', type=list,
                        default=None,
                        help="A list of the runs you want to compare", metavar="compare")
    parser.add_argument('--l', '--latest', type=bool, default=True, 
                         help="If you want to look at the latest run", metavar="latest")
    parser.add_argument('--f', '--folder', type=int, default=None,
                        help="If you want to look at a spesific run", metavar="folder")
    parser.add_argument('--d', '--destination', type=str,
                        default=None, metavar="destination",
                        help='Name of the folder you want the plots to be stored')

    args = parser.parse_args()

    # Check if the choises are legal
    folder_path = path.join(path.dirname(__file__), "..", "nozzle_results", "data")
    if args.f is not None:
        if not path.isdir(path.join(folder_path, str(args.f))): 
            print "The run: %s does not exist." % args.f
            exit(1)
        else:
            args.l = False

    if args.c is not None:
        if isinstance(args.c, list):
            for i in args.c:
                if not path.isfolder(path.join(folder_path, i)):
                    print "The run: %s does not exist." % args.f
                    exit(1)
        else:
            print "Compare is not list"
            exit(1)

    if args.d is not None:
        if not path.isdir(path.join(folder_path, "..", "Plots", args.d)):
            makedirs(path.join(folder_path, "..", "Plots", args.d))

    if args.f is not None and args.c is not None:
        print "You cant provide a list of runs to compare and a spesific run"
        exit(1)

    if args.l and args.c is not None and args.f is not None:
        print "If you want latest you can not spesify another folder or files" \
               + "to compare with"
        exit(1)

    return args.c, args.l, args.f, args.d


def makefolders(filepath):
    # Create new folders for the plots, use same "system" as Oasis
    if filepath is None:
        filepath = path.join(path.dirname(__file__), "..", "nozzle_results", "Plots")

        if not path.exists(filepath):
            filepath = path.join(filepath, "1")
            makedirs(filepath)
        else:
            folders = listdir(filepath)
            newfolder = array([int(folder) for folder in folders]).max() + 1
            filepath = path.join(filepath, str(newfolder))
            makedirs(filepath)
    else:
        filepath = path.join(path.curdir(), filepath)

    return filepath


def main():
    compare, latest, folder, destination = read_command_line()
    data = get_variance(get_data())
    results = get_results(latest=latest, folder=folder, compare=compare)
    filepath = makefolders(destination)
    fancy_plot(results, data, filepath)
    make_plots(results, data, filepath)
    vizualize_flux(results, filepath)
    #compute_validation_matrix(results, filepath)


if __name__ == "__main__":
    main()
