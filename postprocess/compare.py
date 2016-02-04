from os import path, listdir, makedirs
import matplotlib.pyplot as plt
from numpy import asarray, array, linspace
import sys 
import numpy as np
from scipy import stats
from math import sqrt

def get_data():
    """Collect information from the experiments"""
    results_dir = "/home/aslak/master/src/nozzle"
    files = listdir(results_dir)
    map = {}

    # Check all files
    for file in files:
        # Only read files for this case
        if file.split("_")[-2] == "3500":
            tmp = open(path.join(results_dir, file), "r")
            tmp = tmp.readlines()

            # Reset parameters
            line_count = 14
            file_kode = file.split("_")[-1].split(".")[0]

            # Read block by block
            while line_count < len(tmp):
                line = tmp[line_count][:-2].split()
                key = line[0] if len(line) <= 1 else line[0]+line[1]
                number = int(tmp[line_count+1])
                x, u = get_info(tmp[line_count+2:line_count+number+2])

                # Create structure for first file
                if not map.has_key(key):
                    map[key] = {"data": {}, "punkt": {}}
                    
                map[key]["data"][file_kode] = u
                map[key]["punkt"][file_kode] = x

                line_count += number + 2
    return map


def get_info(lines):
    """Help function to exctract data from experiment files"""
    X = []
    U = []
    for line in lines:
        x, u = line.split()
        X.append(float(x))
        U.append(float(u))
    return asarray(X), asarray(U)


def mean_confidence_interval(data):
    """Calculate mean confidence interval (MCI)"""
    a = 1.0*np.array(data)
    n = a.shape[0]
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1+0.95)/2., n-1)
    return m, h, +h


def condifence_interval(data):
    """Extract information for MCI"""
    min = []
    max = []
    m = []
    for i in range(len(data[0])):
        u = []
        for j in range(len(data)):
            if data[j][i] != None:
                u.append(data[j][i])
        if len(u) != 0:
            m_, min_, max_ = mean_confidence_interval(u)
        min.append(min_)
        max.append(max_)
        m.append(m_)

    return m, min, max


def get_variance(map):
    """Linear approximation to compute MCI"""
    tol = 0.00005

    # Different intervals
    x_centerline = [-0.088, -0.064, -0.048, -0.040, -0.016, -0.008, 0, 0.008,
                    0.016, 0.024, 0.032, 0.04, 0.05, 0.06, 0.08]
    pressure_c = [-9.032e-2, -7.696e-2, -6.299e-2, -5.723e-2, -5.149e-2, -4.575e-2,
                  -4.001e-2, -3.0e-2, -1.999E-2  -1.001E-2  -2.03e-3, 0, 2.03e-3, 8.0e-3,
                  1.6e-2, 2.4e-2, 3.2e-2]
    
    #x_centerline = [-0.088, -0.064, -0.048, -0.02, -0.008, 0, 0.008, 0.016,
    #        0.024, 0.032, 0.06, 0.08]
    
    x_r1 = linspace(-0.006+tol, 0.006-tol, 25)
    x_r2 = linspace(-0.002+tol, 0.002-tol, 10)
    x_r3 = linspace(-0.0034+tol, 0.0034-tol, 20)
    new_map = {}

    for key, item in map.iteritems():
        # Pick interval for respective location
        if "pressure" in key:
            print key
            x = pressure_c
        elif "plot-z-distribution" in key:
            x = x_centerline
        elif "-0.02000" in key or "-0.00800" in key or "0.00000" in key:
            x = x_r2
        elif "-0.0480" in key:
            x = x_r3            
        else:
            x = x_r1

        u = []
        for file_kode in map[key]["data"].iterkeys():
            x_data = map[key]["punkt"][file_kode]
            x_data.sort()
            u_data = map[key]["data"][file_kode]
            u.append([])
            for x_punkt in x:
                u_new = None
                for i in range(len(x_data)-1):
                    if x_punkt >= x_data[i] and x_punkt <= x_data[i+1]:
                        vekt_rel = abs(x_data[i] - x_data[i+1])
                        vekt_right = abs(x_data[i]-x_punkt) / vekt_rel
                        vekt_left = abs(x_data[i+1] - x_punkt) / vekt_rel
                        u_new = u_data[i]*vekt_left + vekt_right*u_data[i+1]
                        break
                
                u[-1].append(u_new)

        u, min, max = condifence_interval(u)
        new_map[key] = [u, min, max, x]

    return new_map


def get_data_results(folder_path):
    """Collect stats from a run"""
    stat_path = path.join(folder_path, "Stats")
    data = {"array": {}, "points": {}, "num": 0, "num_initial": 0}
    
    # Get number of evaluations
    data["num"] = 0
    for file in listdir(stat_path):
        if path.isfile(path.join(stat_path, file)):
            if "senterline_u" in file and int(file.split("_")[-1]) > data["num"]:
                data["num"] = int(file.split("_")[-1])

    # Get data
    for file in listdir(stat_path):
        if path.isfile(path.join(stat_path, file)):
            key = "_".join(file.split("_")[:-1])
            num_ = data["num"]
            if "senterline" in key:
                print num_
                print folder_path, key
            arr = np.load(path.join(stat_path, file)) #* num_
            data["array"][key] = arr

    # Get eval points
    for file in listdir(path.join(stat_path, "Points")):
        data["points"][file] = np.load(path.join(stat_path, "Points", file))

    return data


def get_results(latest, folder, compare):
    """Collect data from the computations"""
    folder_path = path.join(path.dirname(__file__), "..", "nozzle_results", "data")

    # Find latest run
    if latest:
        folders = listdir(folder_path)
        folder = array([int(f) for f in folders]).max()

    # Get data from latest or choosen folder
    if compare is None:
        data = get_data_results(path.join(folder_path, str(folder)))

    # Get data from all the runs
    else:
        data = {}
        i = 0
        for folder in compare:
            data[i] = get_data_results(path.join(folder_path, folder))
            i += 1

    return data


def map_filenames(nozzle_header):
    """Map the naming convensions and dissreagard unused data"""
    if "reynolds-stress" in nozzle_header or "jet-width" in nozzle_header \
            or "shear-stress" in nozzle_header:
        return None, None

    if "pressure" in nozzle_header or "shear-stress" in nozzle_header:
        element = 0
    elif "radial" in nozzle_header:
        element = 1
    elif "axial" in nozzle_header:
        element = 2
    
    nozzle_header = nozzle_header.replace("plot-profile-radial-velocity-at-z","slice_u_r")
    nozzle_header = nozzle_header.replace("plot-profile-shear-stress-at-z", "slice_ss_")
    nozzle_header = nozzle_header.replace("plot-wall-distribution-pressure", "wall_p")
    nozzle_header = nozzle_header.replace("plot-profile-axial-velocity-at-z", "slice_u_")
    nozzle_header = nozzle_header.replace("plot-z-distribution-pressure", "senterline_p")
    nozzle_header = nozzle_header.replace("plot-z-distribution-axial-velocity", "senterline_u")

    nozzle_header = nozzle_header.replace("0000", "")
    nozzle_header = nozzle_header.replace("000", "")
    nozzle_header = nozzle_header.replace("00", "")
    nozzle_header = nozzle_header.replace("0.8", "0.008")

    return nozzle_header, element


def make_plots(results, data, filepath, legend, d):
    """Match experimental data against numerical"""
    color = ["r", "g", "c", "k", "b"]

    if 0 in results.keys():
        comp_list = results.keys()
    else:
        comp_list = [0]
        results_ = {}
        results[0] = results

    for key in data.keys():
        key_re, element = map_filenames(key)
        if key_re is not None and "slice_u_r" not in key_re and "deleted" not in key_re:
            plt.figure(figsize=(15, 10))
            #plt.figure()
            if "senterline_u" in key_re:
                plt.title("Time averaged axial velocity at centerline",
                          fontsize=25)
            else:
                plt.title(key)
            u = data[key]
            u[-1] = [u_ * 1000 for u_ in u[-1]]

            #if "senterline_u" in key_re: ###
            tmp_u = d[key]["data"]
            tmp_z = d[key]["punkt"]
            first = True
            tmp_x = {}

            for k in tmp_u.keys():
                for i, p in enumerate(tmp_z[k]):
                    if not tmp_x.has_key(p):
                        tmp_x[p] = [tmp_u[k][i]]
                    else:
                        tmp_x[p].append(tmp_u[k][i])
                    
            #for k, v in tmp_x.items():
            #    u = mean_confidence_interval(v)
            #    k = [(k*1000).tolist()]
            #    y = [u[0].tolist()]
            #    err1 = [u[1].tolist()]
            #    err2 = [u[2].tolist()]
            #    if err1[0] + err2[0] > 10 and "pressure" not in key:
            #        err1 = [0]
            #        err2 = [0]
            #    if first:
            #        plt.errorbar(k, y, yerr=[err1, err2], fmt='bo', label="Data")
            #        first = False
            #    else:
            #        plt.errorbar(k, y, yerr=[err1, err2], fmt='bo')
            #    plt.hold("on")

            plt.errorbar(u[-1], u[0], yerr=[u[1], u[2]], fmt='o', label="Data",
                    linewidth=2)
            plt.hold("on")

            for k in comp_list:
                if key_re.split("_")[-1] == "p":
                    u = results[k]["array"][key_re]*1056 + 1000
                else:
                    u = results[k]["array"][key_re]
                # Convert to mm
                x = results[k]["points"]["_".join(key_re.split("_")[::2])]*1000
                if "slice" in key_re:
                    x = array([x_[0] for x_ in x])
                else:
                    x = array([x_[-1] for x_ in x])
                if element != 0:
                    u = array([u_[element] for u_ in u])
                    plt.plot(x, u, color=color[k], linewidth=2)
                else:
                    plt.plot(x, u[:,0], color=color[k], linewidth=2)
            if legend is not None:
                pass
                #plt.legend(["Data"] + legend, loc=1)
            else:
                plt.legend(["Experiments", "Computational"])
            plt.ylabel(r"$\overline{w}$ [$\frac{m}{s}$]", fontsize=20)
            plt.xlabel(r"$z$ [mm]", fontsize=20)
            #plt.xlim([-0.1, 100])
            #plt.ylim([-0.6, 6])
            #plt.plot([-200, 200], [0, 0], color="k")
            #plt.plot([0, 0], [-1, 6], color="k")
            
            plt.savefig(path.join(filepath, key_re + ".png"))
            #plt.show()
            plt.close()
        else:
            continue
