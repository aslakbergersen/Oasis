from dolfin import *
import numpy as np
from os import path

# Get mesh and coordinates
mesh = Mesh(path.join(path.dirname(path.abspath(__file__)), "..", "mesh", "5M.xml.gz"))
coor = mesh.coordinates()

# Mesh sizes
D = 0.012
D_new = 0.009
L = D
end = 0.2
start = end - L

# Get relevant parts
index = np.nonzero(coor[:,2] >= start)
c = coor[index[0], :]

# Compute factor to move mesh
p = (c[:, 2] - start)/ L
factor = (1 - p) + p * (D_new/D)
#factor = (np.sqrt(c[:,0]**2 + c[:,1]**2) / (D/2)) + ((c[:, 2] - start)/(L)) * D_new/D

# Get new coordinated
new_coor = np.zeros(c.shape)
new_coor[:,0] = c[:,0]*factor
new_coor[:,1] = c[:,1]*factor
new_coor[:,2] = c[:,2]
coor[index[0], :] = new_coor

# Write mesh
file = File("new_mesh.pvd")
file << mesh
