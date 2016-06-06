from argparse import ArgumentParser
from matplotlib.pyplot import *
import numpy as np
from os import path
import scipy.signal as signal
from plot_probes import *
from make_spheres import *
import vtk


if __name__ == "__main__":
    # Get names of cases
    base = path.join(path.dirname(path.abspath(__file__)), "..", "nozzle_results",
                         "data", "%s", "Stats", "Probes")

    # List to store data
    u = []
    t = []
    p = []
    leg = ["5M", "10M", "17M", "28M", "10M P2-P1", "Re=500", "Re=6500"]

    # Chose cases to analyse
    analyce = range(4)
    for i in analyce:
        b = base % ("9" + str(i))
        u_, p_, points, t_ = plot_probes(b, no_plot=True)
        p.append(p_)
        u.append(u_)
        t.append(t_)

    # Get relevant data
    leg_ = [leg[i] for i in analyce]
    index = [(tmp >= 1.5) * (tmp <= 1.6) for tmp in t]
    index[2] = (t[2] >= 1.1) * (t[2] <= 1.2)
    index[3] = (t[3] >= 1.1) * (t[3] <= 1.2)

    # Chose points
    to_check = [-0.1, -0.08, -0.064, -0.056, -0.048, -0.04, -0.04, -0.032,
                -0.024, -0.016, -0.008, 0.0, 0.004, 0.008, 0.012, 0.016, 0.02, 0.024, 0.028,
                0.032, 0.04, 0.044, 0.048, 0.056, 0.06, 0.064, 0.068,
                0.076, 0.08, 0.084, 0.088, 0.092, 0.096, 0.1, 0.12, 0.14, 0.16]

    for i in range(len(analyce)):
        print i
        p_mean = []
        for tmp_point in to_check:
            n = 0
            for k in points:
                if k[2] == tmp_point and k[0] == 0: 
                    p_mean.append(np.mean(p[i][n][0, index[i]])*1000)
                    break
                n += 1

        plot(to_check, p_mean, label=leg[i])
        hold("on")

    title("Pressure loss", fontsize="large")
    xlabel(r"$z$ [$m$]", fontsize="large")
    ylabel(r"pressure [$Pa$]", fontsize="large")
    legend()
    savefig("pressure_loss/pressure_loss.png")
    #show()
    close()
