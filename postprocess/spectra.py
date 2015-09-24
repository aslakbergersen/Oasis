from argparse import ArgumentParser
from matplotlib.pyplot import *
import numpy as np
import scipy.signal as signal
from plot_probes import *
import vtk


def read_command_line():
    """Read arguments from commandline"""
    parser = ArgumentParser()

    # Define arguments
    parser.add_argument('--path', type=str, default="../nozzle_results/5M_P1P1/data/5/Stats/Probes",
                        help="A list of the runs you want to compare",
                        metavar="compare")

    args = parser.parse_args()

    return args.path


def compute_spectra(u_, point, counter):
    show()
    leg = ["5M", "10M", "17M", "28M", "10M-P2P1"]

    fig, ax1 = subplots()
    title("Energy spectrum at z = %s, x = %s" % (point[2], point[0]))
    y_fig = [1, 1, 34, 34, 1, 1]
    x_fig = [-0.12, -0.066, -0.04, 0, 0, 0.2]
    lab = {0: "u", 1: "v", 2: "w"}
    for i in range(1):
        for j in range(len(u_)):
            u = np.asarray(u_[j])[2,:].T

            # Compute mean
            u = np.array(u)
            u_mean = np.mean(u)
            #print "Point z: %s    u_mean = %s" % (str(point[2]), str(u_mean))
			
            # Compute rms values
            #u_rms = ((np.sum((u - u_mean)**2) / len(u))**0.5)
            #print u_rms
				
            # Compute u',v',w' normalize
            u_prime = np.sqrt((u - u_mean)**2)
            #print u_prime
            #u_prime = np.sqrt(((u - u_mean) / u_rms)**2)

            # Compute energy spectra using Welch's method
            fs = 10000.                      # nyquist f of 1000
            window = 'hann'                 # hanning window
            nperseg = int(len(u_prime) / 4.)  # 4 segements
            noverlap = (nperseg / 2)          # 50% overlap
            scaling = 'density'              # scaling has unit V**2/Hz

            # Frequency and energy spectra
            f, E = signal.welch(u_prime, fs=fs, window=window, nperseg=nperseg, 
                                noverlap=noverlap, scaling=scaling)
   
            #print u_mean
            #f = f * 0.012 / u_mean
            #E = E * u_mean / 0.012

            ax1.loglog(f, E, label=leg[j])
            hold("on")
    #plot([10, 100], [1, 10**(-5/3.)])

    #ax2 = ax1.twiny()
    #ax2.plot(x_fig, y_fig, linewidth=3, color='k')
    #ax2.plot([point[2], point[2]], [1, 100], color="r")
    xlabel(r"$f$ [$\frac{1}{s}$]")
    ylabel(r"$E(f)$ [$\frac{m^2}{s}$]")
    ax1.legend()
    ax1.axis([0.1, 10000, 1e-12, 1])
    #ax2.axis([-0.12, 0.2, 1e-22, 100])
    #ax2.axis('off')
    #savefig("movie/spectra_%02.00f.png" % counter)
    #plot(u_, range(len(u_)))
    if point[2] < 0:
        savefig("spectra/spectra_z_minus_%02.04f.png" % point[2])
    else:
        savefig("spectra/spectra_z_%02.04f.png" % point[2])
    #show()
    close()

def plot_spectra(u):
    figure()
    psd(u[:,2], NFFT=u.shape[0], pad_to=u.shape[0]/3, Fs=10000)
    show()

def WritePolyData(input, filename):
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInput(input.GetOutput())
    writer.Write()

def write_spheres(points):
    counter = 0
    for point in points:
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(point)
        sphere.SetRadius(0.0005)

        WritePolyData(sphere, "spheres/" + "sphere%s.vtp" % counter)
        counter += 1


if __name__ == "__main__":
    #case = "P0252"
    #n_points = 6
    #base = "logs/bobo_ipcs_ab_cn_%s_%s_constant_ts10000_cycles2_uOrder1_restart_"
    base = "../nozzle_results/data/%s/Stats/Probes/"
    b_90 = base % "90"
    b_91 = base % "91"
    b_92 = base % "92"
    b_93 = base % "93"
    b_94 = base % "94"

    folder_path = read_command_line()
    u_90, p_90, points = plot_probes(b_90, no_plot=True)
    u_91, p_91, points = plot_probes(b_91, no_plot=True)
    u_92, p_92, points = plot_probes(b_92, no_plot=True)
    u_93, p_93, points = plot_probes(b_93, no_plot=True)
    u_94, p_94, points = plot_probes(b_94, no_plot=True)
    #u_93, p_93, points = plot_probes(b_93, no_plot=True)

    #name_normal = base % (case, "ext")
    #u, u_mag_n, p, time_step = get_data(name_normal, n_points)
    #plot_spectra(u[0])
    #n = 65
    #print points[n]
    #print points[n+1]
    #print points[n-1]
    #counter = 0
    #for p in points:
    #    print p, counter
    #    counter += 1
    #write_spheres(points[56: 84])
    counter = 0
    for n in range(56, 84, 1):
        compute_spectra([u_90[n], u_91[n], u_92[n], u_93[n], u_94[n]], points[n], counter)
        counter += 1
