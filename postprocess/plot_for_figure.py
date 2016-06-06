from os import path, listdir, makedirs
import numpy as np
from argparse import ArgumentParser
from matplotlib.pyplot import *
from plot_probes import sort, plot_probes


def plot_u(u, points, leg, t):
    # Get index of chosen points
    z = ["-0.024", "-0.016", "0.0", "0.016", "0.032", \
         "0.044", "0.056", "0.06", "0.08", "0.12", "0.16"]
    p = []
    for i in range(len(points)):
        if points[i][0] == 0 and points[i][1] == 0:
            if str(points[i][2]) in z:
                p.append(i)

    # Number of cases
    N = len(leg)

    # 'Global' max(t_min) and min(t_max)
    t_min = max([t[i].min() for i in range(N)])
    t_max = min([t[i].max() for i in range(N)])

    # dt for each case
    dt = [t[i][1] - t[i][0] for i in range(N)]

    # Set with of sampling range
    t_diff = t_max - t_min
    #print t_diff
    if t_diff >= 0.1:
        #t_min = t_max - 0.1
        t_min = 1.0
        t_max = 1.1
        #pass
    else:
        print "t_diff to low, please update the data", t_diff
        sys.exit(0)

    # Time array for ploting
    n_steps = [int(round((t_max - t_min) / dt[i])) for i in range(N)]
    t_ = [np.linspace(t_min, t_max, n_steps[i]) for i in range(N)]

    # Set limits for ploting
    limits = []
    for i in range(N):
        if t_[i][-1] == t[i][-1]:
            limits.append([t[i].shape[0] - t_[i].shape[0], t[i].shape[0]])
        else:
            ndiff = (t[i].max() - t_[i].max()) / dt[i]
            limits.append([t[i].shape[0] - ndiff - t_[i].shape[0], t[i].shape[0] - ndiff])

    for index in p:
        figure()
        point = "at z = %s, x = %s" % (points[index][2], points[index][0])

        for i in range(N):
            u_ = u[i][index].T
            u_ = u_[limits[i][0]:limits[i][1]]

            u_mean = np.mean(u_, axis=0)
            k = 0.5*np.sum((u_ - u_mean)**2, axis=1)
            #u_mean_mag = np.sqrt(np.sum(u_mean**2))
            #to_plot = u_mag - u_mean_mag
            plot(t_[i], k, label=leg[i])
            hold("on")
        title("Turbulent kinetic energy at z = %s, x = %s" % (points[index][2],
                points[index][0]), fontsize="x-large")
        xlabel(r"$t$ [s]", fontsize="large")
        ylabel(r"$k$ [$\frac{m^2}{s^2}$]", fontsize="large")
        axis([t_min, t_max, 0, 1])
        #legend()
        savefig("spectra/velo_fluct_" + point.replace(" ", "_") + ".png")
        close()

        for i in range(N):
            u_ = u[i][index].T
            u_ = u_[limits[i][0]:limits[i][1]]
            u_mag = np.sqrt(np.sum(u_**2, axis=1))
            plot(t_[i], u_mag, label=leg[i])
            hold("on")
        title("Velocity magnitude at z = %s, x = %s" % (points[index][2],
               points[index][0]), fontsize="x-large")
        
        xlabel(r"$t$ [$s$]", fontsize="large")
        ylabel(r"|$u(t)$| [$\frac{m}{s}$]", fontsize="large")
        axis([t_min, t_max, 0, 5])
        #legend()
        savefig("spectra/velo_mag_" + point.replace(" ", "_") + ".png")
        close()


if __name__ == "__main__":
    u = []; p = []; t = []
    for folder in ["90", "91", "92", "93"]: #, "94"]:
        print "Loding ...", folder
        folder_path = "../nozzle_results/data/%s/Stats/Probes/" % folder
        u_, p_, points, t_ = plot_probes(folder_path, True)
        u.append(u_)
        p.append(p_)
        t.append(t_)

    plot_u(u, points, ["5M", "10M", "17M", "28M"], t)
