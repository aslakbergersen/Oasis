from os import path, listdir
from scitools.std import plot
from calculate_flux import *
from numpy import asarray, array

def get_data():
    results_dir = "/home/aslak/master/src/nozzle"
    files = listdir(results_dir)
    map = {}

    # Check all files
    for file in files:
        # Only read files for this case
        if file.split("_")[-2] == "6500" and file.split("_")[-1] == "243.txt":
            tmp = open(path.join(results_dir, file), "r")
            tmp = tmp.readlines()
            
            # Reset parameters
            line_count = 14 # check this
            
            # Read block by block
            while line_count < len(tmp):
                line = tmp[line_count][:-2].split()
                key = line[0] if len(line) <= 1 else line[0]+line[1]
                number = int(tmp[line_count+1])
                x, u = get_info(tmp[line_count+2:line_count+number+2])

                # Check length
                if map.has_key(key):
                    if len(u) == len(map[key][0]):
                        map[key] = [map[key][0] + u, map[key][1], map[key][-1] + 1]

                    # Check if it has key
                    else:
                        print "Different length on ", key, len(u), len(map[key][0])

                # Expected trouble
                else:
                    map[key] = [u, x, 1]

                line_count += number + 2

    return map


def get_info(lines):
    X = []
    U = []
    for line in lines:
        x, u = line.split()
        X.append(float(x))
        U.append(float(u))
    return asarray(X), asarray(U)


line_name_map = {
                 "lineX1": "plot-z-distribution-axial-velocity",
                 "lineX2": "plot-profile-axial-velocity-at-z-0.08800",
                 "lineX3": "plot-profile-axial-velocity-at-z-0.06400",
                 "lineX4": "plot-profile-axial-velocity-at-z-0.04800",
                 "lineX5": "plot-profile-axial-velocity-at-z-0.02000",
                 "lineX6": "plot-profile-axial-velocity-at-z-0.00800",
                 "lineX7": "plot-profile-axial-velocity-at-z0.00000",
                 "lineX8": "plot-profile-axial-velocity-at-z0.00800",
                 "lineX9": "plot-profile-axial-velocity-at-z0.01600",
                 "lineX10": "plot-profile-axial-velocity-at-z0.02400",
                 "lineX11": "plot-profile-axial-velocity-at-z0.03200",
                 "lineX12": "plot-profile-axial-velocity-at-z0.06000",
                 "lineX13": "plot-profile-axial-velocity-at-z0.08000",
                 "lineX14": "Inlet",
                 "lineX15": "Outlet"}

sets_path = path.join(path.dirname(__file__), "postProcessing", "sets")
folders = listdir(sets_path)

if len(folders) == 1:
    folder = folders[0]
else:
    fo = [int(f) for f in folders]
    folder = np.asarray(fo).max()

files = listdir(path.join(sets_path, str(folder)))
map = get_data()

for file in files:
    if file.endswith("UMean.xy"):
        U, interval = readfile(path.join(sets_path, str(folder), file))
        key = file.split("_")[0]
        if key not in ["lineX13", "lineX14", "lineX1"]:
            u = map[line_name_map[key]]
       
            new_u = []
            new_x = []
            # Only interested in y > 0 from data:
            for i in range(len(u[0])):
                if u[1][i] >= 0 and u[1][i] <= 0.006:
                    new_u.append(u[0][i])
                    new_x.append(u[1][i])

            new_u = asarray(new_u)/u[-1]
            new_x = asarray(new_x)
            plot(U[1], interval[0], new_x, new_u, title=line_name_map[key])
            raw_input()
        elif key == "lineX1":
            u = map[line_name_map[key]]
            plot(U[0], interval[0], u[1], u[0]/u[-1])
            raw_input()
