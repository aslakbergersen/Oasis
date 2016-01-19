from argparse import ArgumentParser
from matplotlib.pyplot import *
from matplotlib import mlab
import numpy as np
from os import path
import scipy.signal as signal
from plot_probes import *
from make_spheres import *
import vtk


def compute_spectra(u_, point, fs, leg):
    for i in range(len(u_)):
        u = np.asarray(u_[i]).T

        # Get turbulent kinetic energy signal
        u = np.array(u)
        u_mean = np.mean(u, axis=0)
        #u_prime = np.sqrt(np.sum((u - u_mean)**2, axis=1))
        u_prime = 0.5*np.sum((u - u_mean)**2, axis=1)

        #E, f = mlab.psd(u_prime, NFFT=int(u_prime.shape[0] / 8), Fs=fs[i], detrend=mlab.detrend_none,
        #    window=mlab.window_hanning, noverlap=int(u_prime.shape[0] / 16), pad_to=None,
        #    sides='default', scale_by_freq=None)

        # Compute energy spectra using Welch's method
        window = 'hanning'                # hanning window (default)x
        nperseg = int(u_prime.shape[0] / 8) #8192                    # 4 segements (256 default)
        noverlap = None #int(nperseg / 2) # 50% overlap (default)
        #scaling = 'density'               # density = power spectral density [V**2 / Hz] (default)
        scaling = 'spectrum'              # spectrum = power spectrum  [V**2]

        # Frequency and energy spectra
        f, E = signal.welch(u_prime, fs=fs[i], window=window, nperseg=nperseg, 
                            noverlap=noverlap, scaling=scaling)
   
        # St. number scaling
        #f = f * 0.012 / np.mean(u_mean) / 2
        #E = E * u_mean / 0.012

        loglog(f, E, label=leg[i])
        hold("on")


    title("Power spectral density at z = %s, x = %s" % (point[2], point[0]),
            fontsize="x-large")
    xlabel(r"$f$ [$\frac{1}{s}$]", fontsize="large")
    ylabel(r"PSD [$\frac{m^2}{s}$]", fontsize="large")
    #legend()
    axis([10, 10000, 1e-14, 1])

    if point[2] < 0:
        savefig("spectra/spectra_z_minus_%02.04f.png" % point[2])
    else:
        savefig("spectra/spectra_z_%02.04f.png" % point[2])
    #show()
    close()




if __name__ == "__main__":
    # Test case
    """
    fs= 10000
    x = np.linspace(0, 2, fs)
    u = 100*np.sin(2*1000* x * np.pi) #+ np.sin(10 * x * np.pi) + np.sin(100 * x * np.pi)
    noise_power = 5
    u += np.random.normal(scale=np.sqrt(noise_power), size=x.shape)
    compute_spectra([u], [0,1,10], [fs/2], "Test")
    sys.exit(0)
    """
    
    # Get names of cases
    base = path.join(path.dirname(path.abspath(__file__)), "..", "nozzle_results",
                         "data", "%s", "Stats", "Probes")

    # List to store data
    u = []
    t = []
    leg = ["5M", "10M", "17M", "28M", "10M P2-P1", "Re=500", "Re=6500"]

    # Chose cases to analyse
    analyce = range(4)
    #analyce = [6] #[0, 2] #[6]
    for i in analyce:
        b = base % ("9" + str(i))
        u_, p, points, t_ = plot_probes(b, no_plot=True)
        t.append(t_)
        u.append(u_)

    # Get relevant data
    leg_ = [leg[i] for i in analyce]
    fs = [1/(tmp[1] - tmp[0]) for tmp in t]
    index = [(tmp >= 1.0) * (tmp <= 1.2) for tmp in t]

    # Chose points
    for p in [0.0, 0.012, 0.016, 0.02, 0.032, 0.044, 0.08, 0.12, 0.16,]:
        n = 0
        for k in points:
            if k[2] == p and k[0] == 0: break
            n += 1

        # Plot PSD
        u_ = [u_[n] for u_ in u]
        u_tmp = [u_[i][:, index[i]] for i in range(len(index))]
        compute_spectra(u_tmp, points[n], fs, leg_)
