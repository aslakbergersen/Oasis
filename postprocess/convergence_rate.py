from os import listdir, path
import numpy as np
import re
from math import log, sqrt
from matplotlib.pyplot import *
import sys

if len(sys.argv) > 1:
    folder = sys.argv[1]
else:
    print "Usage: python %s folder" % sys.argv[0]
    print "Assume latest run"
    abs_path = path.dirname(path.abspath(__file__))
    result_path = path.join(abs_path, "..", "output", "MMS")
    files = listdir(result_path)
    folder = max([int(i) for i in files])
    #sys.exit(0)

abs_path = path.dirname(path.abspath(__file__))
result_path = path.join(abs_path, "..", "output", "MMS", str(folder))
files = listdir(result_path)

h = []
dt = []
E_u0 = []
E_u1 = []
E_u2 = []
E_p = []

files.sort(key=lambda x: (float(x.split("_")[1]), -float(x.split("_")[-2])))
print files
for file in files:
    f = open(path.join(result_path, file), "r")
    text = f.read()
    f.close()
    dt.append(float(file.split("_")[-2])) 
    h.append(float(re.search(r"hmin: (.*)", text).groups()[0]))
    E_p.append(eval(re.search(r"'p': (\[.*?\])", text).groups()[0]))
    E_u0.append(eval(re.search(r"'u0': (\[.*?\])", text).groups()[0]))
    E_u1.append(eval(re.search(r"'u1': (\[.*?\])", text).groups()[0]))
    E_u2.append(eval(re.search(r"'u2': (\[.*?\])", text).groups()[0]))

# Last error
step = -1
if h[-1] / h[-2] == 1:
    r_u0 = [log(E_u0[i-1][step] / E_u0[i][step]) / log(dt[i-1] / dt[i]) for i in range(1, len(E_u0))]
    r_u1 = [log(E_u1[i-1][step] / E_u1[i][step]) / log(dt[i-1] / dt[i]) for i in range(1, len(E_u1))]
    r_u2 = [log(E_u2[i-1][step] / E_u2[i][step]) / log(dt[i-1] / dt[i]) for i in range(1, len(E_u2))]
    r_p =  [log(E_p[i-1][step] / E_p[i][step]) / log(dt[i-1] / dt[i]) for i in range(1, len(E_p))]
    k = dt
else:
    r_u0 = [log(E_u0[i-1][step] / E_u0[i][step]) / log(h[i-1] / h[i]) for i in range(1, len(E_u0))]
    r_u1 = [log(E_u1[i-1][step] / E_u1[i][step]) / log(h[i-1] / h[i]) for i in range(1, len(E_u1))]
    r_u2 = [log(E_u2[i-1][step] / E_u2[i][step]) / log(h[i-1] / h[i]) for i in range(1, len(E_u2))]
    r_p = [log(E_p[i-1][step] / E_p[i][step]) / log(h[i-1] / h[i]) for i in range(1, len(E_p))]
    k = h

print "u0:", r_u0
print "u1:", r_u1
print "u2:", r_u2
print [E_u1[i][step] for i in range(0, len(E_u2))]
print h
#print [E_p[i][step] for i in range(0, len(E_p))]
#print "mean:", [(r_u0[i] + r_u1[i]) / 2. for i in range(3)]
print "p:", r_p

figure()
for i in range(len(E_u0)):
    t = [dt[i]*10*k for k in range(len(E_u0[i]))][1:]
    semilogy(t, E_u0[i][1:])
    hold("on")
show()

figure()
for i in range(len(E_p)):
    t = [dt[i]*10*k for k in range(len(E_p[i]))][1:]
    semilogy(t, E_p[i][1:])
    hold("on")
show()
