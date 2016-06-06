from matplotlib.pyplot import *
from scipy.ndimage.filters import gaussian_filter
import numpy as np

y = np.load("../nozzle_results/data/90/Stats/tmp/senterline_u_221801")[:, 2]
x = np.load("../nozzle_results/data/90/Stats/Points/senterline")[:, 2]

print y.shape
print x.shape

plot(x, y)

hold("on")

y_smooth = y
for i in range(10000):
    y_smooth = gaussian_filter(y, 40)

#plot(x, y_smooth)
#show()

#plot(x, y_smooth)
#show()

y = np.load("../nozzle_results/data/90/Stats/tmp/senterline_u_221801")
y[:,2] = y_smooth
y.dump("../nozzle_results/data/90/Stats/senterline_u_221801")
