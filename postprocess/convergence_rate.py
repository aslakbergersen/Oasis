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
E_p = []

files.sort(key=lambda x: (float(x.split("_")[1]), -float(x.split("_")[-1][:-4])))

for file in files:
    f = open(path.join(result_path, file), "r")
    text = f.read()
    f.close()
    
    dt.append(float(file.split("_")[-1][:-4])) 
    h.append(float(re.search(r"hmin: (.*)", text).groups()[0]))
    #E_p.append(eval(re.search(r"Error p l2 min: (\[.*\])", text).groups()[0]))
    #E_u.append(eval(re.search(r"Error u l2 min: (\[.*\])", text).groups()[0]))
    E_p.append(eval(re.search(r"'p': (\[.*?\])", text).groups()[0]))
    E_u0.append(eval(re.search(r"'u0': (\[.*?\])", text).groups()[0]))
    E_u1.append(eval(re.search(r"'u1': (\[.*?\])", text).groups()[0]))


# Total error
#E_u_ = [np.sum(np.asarray(E_u[i])) * 1e-8 for i in range(len(E_u))]
#r_up = [log(E_u_[i-1] / E_u_[i]) / log(h[i-1] / h[i]) for i in range(1, len(E_u))]

#print r_up
print E_u0
# Last error
#r_p_dt = [log(E_u[i-1][-1] / E_u[i][-1]) / log(dt[i-1] / dt[i]) for i in range(1, len(E_u))]
#r_u_dt = [log(E_p[i-1][-1] / E_p[i][-1]) / log(dt[i-1] / dt[i]) for i in range(1, len(E_p))]
r_u0 = [log(E_u0[i-1][-1] / E_u0[i][-1]) / log(h[i-1] / h[i]) for i in range(1, len(E_u0))]
r_u1 = [log(E_u1[i-1][-1] / E_u1[i][-1]) / log(h[i-1] / h[i]) for i in range(1, len(E_u1))]
r_p = [log(E_p[i-1][1] / E_p[i][1]) / log(h[i-1] / h[i]) for i in range(1, len(E_p))]

print r_u0
print r_u1
print r_p
#print r_u_dt
#print r_p_dt

#print E_u
#e = np.asarray(E_u)
#print e[:,-1]
#print r_u
#print r_p

dt = 1e-8
t = [dt*i for i in range(len(E_u[0])-1)]
l = ["%.02e" % h[i] for i in range(len(h))]
#print l
#print E_u
#print E_p

figure()
for i in range(len(h)):
    #print t
    #print E_u[i]
    semilogy(t, E_u[i][1:])
    #plot(t, E_u[i][1:])
    hold("on")
legend(l)
show()
#print E_p[-1]

figure()
for i in range(len(h)):
    #print t
    #print E_u[i]
    semilogy(t, E_p[i][1:])
    hold("on")
legend(l)
show()

#print E_u
#print E_p
#print "Velocity:", r_u
#print "Pressure", r_p
