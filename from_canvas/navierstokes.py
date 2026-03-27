import os
import sys

if sys.platform.startswith("win"):
    dll_dirs = [
        os.path.join(sys.prefix, "bin"),
        os.path.join(sys.prefix, "Library", "bin"),
        os.path.join(sys.prefix, "Lib", "site-packages", "netgen"),
    ]
    for dll_dir in dll_dirs:
        if os.path.isdir(dll_dir):
            os.add_dll_directory(dll_dir)
            os.environ["PATH"] += os.pathsep + dll_dir

from ngsolve import *
from netgen.csg import *
import numpy as np
import matplotlib.pyplot as plt

try:
    # Notebook/web-friendly visualization backend.
    from ngsolve.webgui import Draw as NGS_Draw
except ImportError:
    NGS_Draw = Draw

try:
    # Enables native Netgen GUI windows when available.
    import netgen.gui
except ImportError:
    netgen = None


# viscosity
nu = 0.001

# timestepping parameters
tau = 0.001
tend = 10

from netgen.geom2d import SplineGeometry
geo = SplineGeometry()
geo.AddRectangle( (0, 0), (2, 0.41), bcs = ("wall", "outlet", "wall", "inlet"))
geo.AddCircle ( (0.2, 0.2), r=0.05, leftdomain=0, rightdomain=1, bc="cyl", maxh=0.02)
mesh = Mesh( geo.GenerateMesh(maxh=0.07))

mesh.Curve(3)


for e in mesh.edges:
    line = np.array([mesh.vertices[v.nr].point for v in e.vertices])
    plt.plot(line[:,0],line[:,1],c='gray',alpha=0.75)
for v in mesh.vertices:
    plt.text(*v.point,v,color='red')
plt.gca().set_axis_off()
plt.gca().set_aspect(1)
plt.show()

V = VectorH1(mesh,order=3, dirichlet="wall|cyl|inlet")
Q = H1(mesh,order=2)

X = V*Q

u,p = X.TrialFunction()
v,q = X.TestFunction()

stokes = nu*InnerProduct(grad(u), grad(v))+div(u)*q+div(v)*p - 1e-10*p*q
a = BilinearForm(X, symmetric=True)
a += stokes*dx
a.Assemble()

# nothing here ...
f = LinearForm(X)   
f.Assemble()

# gridfunction for the solution
gfu = GridFunction(X)

# parabolic inflow at inlet:
uin = CoefficientFunction( (1.5*4*y*(0.41-y)/(0.41*0.41), 0) )
gfu.components[0].Set(uin, definedon=mesh.Boundaries("inlet"))

# solve Stokes problem for initial conditions:
inv_stokes = a.mat.Inverse(X.FreeDofs())

res = f.vec.CreateVector()
res.data = f.vec - a.mat*gfu.vec
gfu.vec.data += inv_stokes * res


# matrix for implicit Euler 
mstar = BilinearForm(X, symmetric=True)
mstar += (u*v + tau*stokes)*dx
mstar.Assemble()
inv = mstar.mat.Inverse(X.FreeDofs(), inverse="sparsecholesky")

# the non-linear term 
conv = BilinearForm(X, nonassemble = True)
conv += (grad(u) * u) * v * dx

# for visualization
scene = NGS_Draw(Norm(gfu.components[0]), mesh, "velocity", sd=3)

# implicit Euler/explicit Euler splitting method:
t = 0
with TaskManager():
    while t < tend:
        print ("t=", t, end="\r")

        conv.Apply (gfu.vec, res)
        res.data += a.mat*gfu.vec
        gfu.vec.data -= tau * inv * res    

        t = t + tau
        Redraw()


