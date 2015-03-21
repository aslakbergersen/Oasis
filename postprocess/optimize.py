from os import path, listdir
import re

path_folders = path.join(path.dirname(__file__), "..", "output")
cases = listdir(path_folders)


for case in cases:
    print case
    data = []
    tests = listdir(path.join(path_folders, case))
    for test in tests:
	print test
	dt = []
        tmp, mpi, deg = test.split("_")  # Split filename
        mpi = mpi[2:]                    # Remove mpi from number of threads
        deg = deg.split(".")[0]          # Remove .out
        text = open(path.join(path_folders, case, test), "r").read()
        a = re.findall(r"previous\s(.*)\stimesteps\s=\s(.*)\x1b\[0m", text)
        for nstep, time in a:
            dt.append(float(time) / float(nstep))

	print dt
