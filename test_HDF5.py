from dolfin import *
from numpy import *

newfile = HDF5File(mpi_comm_world(), "test.h5", "w")
