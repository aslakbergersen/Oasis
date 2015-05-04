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
E_u = []
E_p = []

print files
files.sort(key=lambda x: (float(x.split("_")[1]), -float(x.split("_")[-1][:-4])))
print files

for file in files:
    f = open(path.join(result_path, file), "r")
    text = f.read()
    f.close()
    #print text
    #sys.exit(0)
    #dt.append(1e-6)  #float(re.search(r"dt: (.*)", text).groups()[0]))
    h.append(float(re.search(r"hmin: (.*)", text).groups()[0]))
    E_p.append(eval(re.search(r"Error p l2 min: (\[.*\])", text).groups()[0]))
    E_u.append(eval(re.search(r"Error u l2 min: (\[.*\])", text).groups()[0]))

r_u = [log(E_u[i-1][-1] / E_u[i][-1]) / log(h[i-1] / h[i]) for i in range(1, len(E_u))]
r_p = [log(E_p[i-1][1] / E_p[i][1]) / log(h[i-1] / h[i]) for i in range(1, len(E_p))]

#print E_u
#e = np.asarray(E_u)
#print e[:,-1]
print r_u
print r_p

dt = 1e-8
t = [dt*i for i in range(len(E_u[0])-1)]
l = ["%.02e" % h[i] for i in range(len(h))]
print l
print E_u
print E_p
figure()
for i in range(len(h)):
    #print t
    #print E_u[i]
    semilogy(t, E_u[i][1:])
    #plot(t, E_u[i][1:])
    hold("on")
legend(l)
show()
print E_p[-1]

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
print "Velocity:", r_u
print "Pressure", r_p
