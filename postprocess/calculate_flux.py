from os import listdir, path
import sys
import matplotlib.pyplot as plt
from math import pi
import numpy as np

# Hardcoded constant for theoretical volumetric flow
# for FDA case Re = 3500
Q = 3.64E-5

def compute_flux(U, y):
    flux = 0
    for i in range(len(U)-1):
        #flux += (U[i+1]+U[i])/2. * (y[i+1]-y[i]) * (y[i+1]+y[i])/2.
        flux += (U[i+1]*y[i+1] + U[i]*y[i]) / 2. * (y[i+1]-y[i])
    return 2*flux*pi#flux*2*pi

def vizualize_flux(results, filepath):
    flux = []
    for key, value in results["array"].items():
        if "slice_u" in key and "slice_u_r" not in key:
            N = results["array"][key][:,2].shape[0] / 2
            arr = results["array"][key][:,2][N:]
            points = results["points"]["_".join(key.split("_")[::2])][:,0][N:]
            f = compute_flux(arr, points)
            flux.append((f, float(key.split("_")[2])))
            print "%03.04f:  %e" % (flux[-1][-1], flux[-1][0])

    flux.sort(key=lambda x: x[-1])
    z = [x[1] for x in flux]
    flux = [abs(x[0]-Q)/Q for x in flux]
    plt.figure()
    plt.plot(z,flux)
    plt.title(r"Flux $\frac{Q_{CFD} - Q_{theory}}{Q_{theory}}$")
    plt.legend("Flux")
    plt.xlabel(r"Q [$\frac{m^3}{s}$]")
    plt.ylabel("z [m]")
    #plt.show()
    plt.savefig(path.join(filepath, "flux.png"))
