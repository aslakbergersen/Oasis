from matplotlib.pyplot import *
from compare import *
from numpy import array
from os import path

# Get data from experiments
map = get_data()
data = get_variance(map)

def mirror(x, neg=False):
    x_tmp = x.tolist()
    x = x.tolist()
    x.reverse()
    if neg:
        x = list(-array(x))
    return x_tmp + x

x = [-1, 0, 2, 4, 6, 8, 15, 20]
edge = array([-0.002, -0.002, -0.006, -0.006])
edge_x = [-4, 0, 1e-10, 21]
D = 0.004
scale = 2
first = True
labels = []

#TODO: Compute a map
line_name_map = {
                 "lineX6": "plot-profile-axial-velocity-at-z-0.00800",
                 "lineX7": "plot-profile-axial-velocity-at-z0.00000",
                 "lineX8": "plot-profile-axial-velocity-at-z0.00800",
                 "lineX9": "plot-profile-axial-velocity-at-z0.01600",
                 "lineX10": "plot-profile-axial-velocity-at-z0.02400",
                 #"lineX11": "plot-profile-axial-velocity-at-z0.03200",
                 "lineX12": "plot-profile-axial-velocity-at-z0.06000",
                 "lineX13": "plot-profile-axial-velocity-at-z0.08000",
                }

# Plot walls
#figure()
fig, ax = plt.subplots()
plot(edge_x, edge, edge_x, -edge, linewidth=3, color='k')
ylim((-0.0065, 0.0065))
xlim((-4, 21))
hold("on")
ax.set_xticks([-1, 2,4, 6, 15, 20])

# Plot arrows
#arrow(-3, 0, 0, 0.002, fc='k', ec='k')
#arrow(-3, 0, 0, -0.002, fc='k', ec='k')

# Set attributs to plot
title("Aksial hastighet ved flere slicer")
xlabel("x/D [-]")
ylabel("r [m]")
#p.set_xticks([-1, 0, 2, 8, 15 ,20])

# Choose experiments
locations = []
locations +=  [path.join("wedge_simple_medium", "postProcessing", "sets", "28.5")]
locations +=  [path.join("omega","wedge_simple_medium", "postProcessing", "sets", "15.9")]

# Plot experimental data
for key, value in line_name_map.iteritems():
    u = data[value]
    displacement = float(value.split("-z")[-1])/D
    x = mirror(array(u[0])/scale + displacement)
    u_ = mirror(array(u[-1]), True)
    if first:
        errorbar(x, u_, xerr=[mirror(array(u[1])/scale),
                 mirror(array(u[2])/scale)], fmt="o", label="Data", color="b")
    else:
        errorbar(x, u_, xerr=[mirror(array(u[1])/scale), 
                 mirror(array(u[2])/scale)], fmt="o", label=None,
                 color="b")

    for l in locations:
        x_, u_ = readfile(path.join(l, key + "_UMean.xy"))
        x = mirror(u_[0]/scale + displacement)
        u = mirror(x_[1], True)
        color = "r" if l.split("/")[0] == "omega" else "g"
        if first:
            label = r"$k-\omega$" if l.split("/")[0] == "omega" else r"$k-\epsilon$"
            tmp, = plot(x, u, color=color, label=label)
        else:
            plot(x, u, color=color, label=None)
    first = False

legend()
#show()
savefig("fancy_plot.png")
