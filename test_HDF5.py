from dolfin import *

newfile = HDF5File(mpi_comm_world(), "test.h5", "w")
newfile.writfrom dolfin import *

newfile = HDF5File(mpi_comm_world(), "test.h5", "w")
newfile.flush()
newfile.write(4, "/current")
newfile.close()
