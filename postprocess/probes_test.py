from fenicstools import *
from dolfin import *
from numpy import *

mesh = UnitSquareMesh(10, 10)
V = VectorFunctionSpace(mesh, "CG", 1)
x = array([[0.1, 0.4], [0.5, 0.2], [0, 0.9]])
ps = Probes(x.flatten(), V)

u = interpolate(Expression(("x[0]", "x[1]")), V)
ps(u)
ps(u)
ps(u)

ps.array(filename="sdfklj")
ps.clear()
ps.array(filename="sdkdk")
