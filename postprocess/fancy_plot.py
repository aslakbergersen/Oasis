from matplotlib.pyplot import *
from compare import *
from numpy import array
from os import path

def fancy_plot(results, data, filepath):
    x = [-1, 0, 2, 4, 6, 8, 15, 20]
    edge = array([-0.002, -0.002, -0.006, -0.006])
    edge_x = [-4, 0, 1e-10, 21]
    D = 0.004
    scale = 2
    labels = []
    first = True

    fig, ax = plt.subplots()
    plot(edge_x, edge, edge_x, -edge, linewidth=3, color='k')
    ylim((-0.0065, 0.0065))
    xlim((-2.5, 21))
    hold("on")
    ax.set_xticks([-2, 0, 2, 4, 6, 8, 15, 20])

    # Set attributs to plot
    title("Aksial hastighet ved flere slicer")
    xlabel("x/D [-]")
    ylabel("r [m]")
    #p.set_xticks([-1, 0, 2, 8, 15, 20])
    
    # Plot experimental data
    for key in data.keys():
        key_re, element = map_filenames(key)
        if key_re is not None and "slice_u" in key_re and "slice_u_r" not in key_re:
            u = data[key]
            displacement = float(key_re.split("_")[-1])/D
            x = array(u[0])/scale + displacement
            u_ = array(u[-1])

            if first:
                errorbar(x, u_, xerr=[array(u[1])/scale, array(u[2])/scale], 
                         fmt="o", label="Data", color="b")
            else:
                errorbar(x, u_, xerr=[array(u[1])/scale, array(u[2])/scale],
                                      fmt="o", color="b")

            x = results["array"][key_re][:,2] / scale + displacement
            u = results["points"]["_".join(key_re.split("_")[::2])][:,0]

            if first:
                tmp, = plot(x, u, color="r", label="CFD")
            else:
                tmp, = plot(x, u, color="r")

            first = False

    legend()
    show()
    #savefig(path.join(filepath, "fancy_plot.png"))
