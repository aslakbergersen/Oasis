from os import path, listdir, makedirs
import numpy as np
from argparse import ArgumentParser
from matplotlib.pyplot import *

def read_command_line():
    """Read arguments from commandline"""
    parser = ArgumentParser()

    # Define arguments
    parser.add_argument('--path', type=str,
                        default="../nozzle_results/5M_P1P1/data/5/Stats/Probes",
                        help="A list of the runs you want to compare", metavar="compare")

    args = parser.parse_args()

    return args.path


def fill_data(files, data, size_file, tstep_interval):
    i = 0
    s_prev = 0
    for file in files:
        a = np.load(file)
        s = size_file if i == 0 else tstep_interval
        print file
        data[:, :, s_prev:s_prev+s] = a
        s_prev += s 
        i += 1
    return data


def sort(list):
    return sorted(list, key= lambda x: int(x.split("_")[-2]))


# TODO: test if matplotlib is on abel
def plot_probes(folder_path, no_plot=False):
    """Plot instantanius velocity profiles at probe points"""
    files = listdir(folder_path)
    if not path.isdir(path.join(folder_path, "..", "Plots")):
        makedirs(path.join(folder_path, "..", "Plots"))
    dest_path = path.join(folder_path, "..", "Plots")

    # Sort the files
    u_files = sort([path.join(folder_path, f) for f in files if "u_" in f])
    p_files = sort([path.join(folder_path, f) for f in files if "p_" in f])
    points = np.load(path.join(folder_path, "points"))

    # Informations on shape of data
    tstep = [int(f.split("_")[-2]) for f in u_files]
    num_nodes, num_dim, size_file = np.load(u_files[0]).shape
    if len(tstep) > 1:
        tstep_interval = tstep[1] - tstep[0]
    else:
        tstep_interval = 0
    num_eval = (len(u_files) - 1) * tstep_interval + size_file
    dt = float(u_files[0].split("_")[-3])

    # Load u in pre allocated data structure
    data_u = fill_data(u_files, np.zeros((num_nodes, num_dim, num_eval)),
                       size_file, tstep_interval)
    data_p = fill_data(p_files, np.zeros((num_nodes, 1, num_eval)), size_file, tstep_interval)
    
    if no_plot:
        return data_u, data_p, points

    # Make plots for each node
    for i in range(num_nodes):
        u = data_u[i].T
        p = data_p[i]
        u_axial = u[:, 2]
        u_mag = np.sqrt(np.sum(u**2, axis=1))
        u_mean = np.mean(u, axis=0)
        u_mean_mag = np.sqrt(np.sum(u_mean**2))
        
        # Time
        first_tstep = tstep[0] - (tstep[1] - tstep[0])
        t = np.linspace(first_tstep, num_eval + first_tstep, num_eval) * dt

        point = "at z = %s, x = %s" % (points[i][2], points[i][0])
        file = "_z%s_x%s.png" % (points[i][2], points[i][0])

        # Plot u axial
        figure()
        plot(t, u_axial)
        title("Axial velocity " + point)
        xlabel("t [s]")
        ylabel("w(t) [m/s]")
        savefig(path.join(dest_path, "probe_w" + file))
        close()

        # Plot u magnitude
        figure()
        plot(t, u_mag)
        title("Velocity magnitude " + point)
        xlabel("t [s]")
        ylabel("|u(t)| [m/s]")
        savefig(path.join(dest_path, "probe_mag" + file))
        close()

        # Plot u' with u_mean
        figure()
        plot(t, u_mag - u_mean_mag)
        #hold("on")
        #tmp = np.zeros(u_mag.shape)
        #tmp[:] = u_mean_mag
        #plot(t, tmp)
        #hold("off")
        #legend(["Fluktuating component", "Mean component"])
        title("Fluctuating component " + point)
        xlabel("t [s]")
        ylabel("u [m/s]")
        savefig(path.join(dest_path, "probe_fluct" + file))
        close()

        # Plot p
        figure()
        plot(t[10:], p.T[10:])
        title("Pressure " + point)
        xlabel("t [s]")
        ylabel("p [N/m^2]")
        savefig(path.join(dest_path, "probe_p" + file))
        close()


if __name__ == "__main__":
    folder_path = read_command_line()
    plot_probes(folder_path)
