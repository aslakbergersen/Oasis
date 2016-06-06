import numpy as np
from os import *
from matplotlib.pyplot import *

arr1 = np.load("nozzle_results/data/6/Stats/senterline_u_5001")
arr2 = np.load("nozzle_results/data/8/Stats/senterline_u_9001")
points = np.load("nozzle_results/data/8/Stats/Points/senterline")

pl1 = arr1[:,2] / 5001
print arr2
pl2 = arr2[:,2] / 25000010
print pl2

#print len(pl1)
#print len(points)
plot(points[:100], pl1[:100], points[:100], pl2[:100])
legend(["5001", "9001"])
axis([-0.12, -0.113, 0.63, 0.64])
show()
print arr1[:, 2] / 5001. - arr2[:, 2] / 25000001
