from dolfin import *
from numpy import *

newfile = HDF5File(mpi_comm_world(), "test.h5", "w")
newfile.flush()
newfile.write(array([4,5]), "/current")
newfile.close()
