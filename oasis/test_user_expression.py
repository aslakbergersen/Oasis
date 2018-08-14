from dolfin import *
from os import path

"""
# User expression
class test(UserExpression):
    def __init__(self, dummy, dummy1, **kwargs):
        self.dummy = dummy
        self.dummy1 = dummy1
        super().__init__(**kwargs)

    def eval(self, value, x):
        value[:] = x[0]
"""

# Param
dt = 0.01
N = 200

# Mesh
#m = UnitSquareMesh(20, 20)
m = RectangleMesh(Point(0.0, 0.0), Point(3.0, 1.0), 20*3, 20)

# Function spaces
V = VectorFunctionSpace(m, "CG", 1)

# Functions
u_mesh, v_mesh = TrialFunction(V), TestFunction(V)
f_mesh = Function(V)
move = Function(V)
d_1 = Function(V)
d_ = Function(V)

# Variational formulation
F = inner(grad(u_mesh), grad(v_mesh))*dx + inner(f_mesh, v_mesh)*dx

# Boundary conditions
left = AutoSubDomain(lambda x, b: b and near(x[0], 0))
right = AutoSubDomain(lambda x, b: b and near(x[0], 3))

boundaries = MeshFunction("size_t", m, m.topology().dim() - 1)
boundaries.set_all(0)
left.mark(boundaries, 1)
right.mark(boundaries, 2)

f = File("test_b.pvd")
f << boundaries

left_ex = Expression(("t", 0), t=0, element=V.ufl_element())

left_bc = DirichletBC(V, left_ex, boundaries, 1)
right_bc = DirichletBC(V, Constant((0, 0)), boundaries, 2)

bc_mesh = [right_bc, left_bc]

# Solver
mesh_prec = PETScPreconditioner("ilu")
mesh_sol = PETScKrylovSolver("gmres", mesh_prec) #, mesh_prec)
w_vec = Function(V)

# Store results
viz_d = XDMFFile(MPI.comm_world, path.join("test_deformation", "VTK", "deformation.xdmf"))
viz_d.parameters["rewrite_function_mesh"] = True
viz_d.parameters["flush_output"] = True

# Time-loop
for i in range(N):
    left_ex.t = i*dt

    # Read deformation
    d_1.vector().zero()
    d_1.vector().axpy(1, d_.vector())

    # Solve for d and w
    A, L = system(F)
    solve(A == L, d_, bc_mesh)
    #assemble(a_mesh, tensor=A_mesh)
    #assemble(l_mesh, tensor=L_mesh)

    #for bc in bc_mesh:
    #    bc.apply(A_mesh, L_mesh)

    #mesh_sol.solve(A_mesh, d_.vector(), L_mesh)
    move.vector().zero()
    move.vector().axpy(1, d_.vector())
    move.vector().axpy(-1, d_1.vector())
    ALE.move(m, move)

    m.bounding_box_tree().build(m)

    viz_d.write(d_, i*dt)
