import netgen.gui
from ngsolve import *
from netgen.occ import *

# parameters: feel free to change these
viscosity = 0.001
inlet_velocity = 1
tend = 5

# largest we got away with is 0.001
timestep = 0.001


# Domain Geometry
rect = Rectangle(2.2, 0.41).Face()
rect.edges.name = "wall"
rect.edges.Min(X).name = "inlet"
rect.edges.Max(X).name = "outlet"

circle_face = Circle(Pnt(0.2, 0.2), 0.05).Face()
circle_face.edges.name = "cyl"

shape = rect - circle_face 

# generate mesh
geo = OCCGeometry(shape, dim=2)
mesh = Mesh(geo.GenerateMesh(maxh=0.15)).Curve(3)

V = VectorH1(mesh, order=3, dirichlet="wall|cyl|inlet")
Q = H1(mesh, order=2, dirichlet="outlet")
X = V * Q

u, p = X.TrialFunction()
v, q = X.TestFunction()

#  combine the weak forms of the momentum and continuity equations
stokes = (viscosity*InnerProduct(grad(u), grad(v)) + div(u) * q + div(v) * p - 1e-10 * p * q) * dx

a = BilinearForm(stokes).Assemble()
f = LinearForm(X).Assemble()

gfu = GridFunction(X)

# boundary conditions
uin = CoefficientFunction((inlet_velocity, 0))
gfu.components[0].Set(uin, definedon=mesh.Boundaries("inlet"))
gfu.components[1].Set(CoefficientFunction(0), definedon=mesh.Boundaries("outlet"))

gfu_bnd = GridFunction(X)
gfu_bnd.components[0].Set(uin, definedon=mesh.Boundaries("inlet"))
gfu_bnd.components[1].Set(CoefficientFunction(0), definedon=mesh.Boundaries("outlet"))

inv_stokes = a.mat.Inverse(X.FreeDofs())
res = f.vec - a.mat * gfu.vec
gfu.vec.data += inv_stokes * res

Draw(gfu.components[0], mesh, "velocity")
print("press enter to start sim")
input()


m = BilinearForm(u * v * dx).Assemble()

# M* = M + dt*A
mstar = BilinearForm(u * v * dx + timestep * stokes).Assemble()

inv = mstar.mat.Inverse(X.FreeDofs()) 

conv = BilinearForm(X, nonassemble=True)
conv += (Grad(u) * u) * v * dx

t = 0
i = 0
print("starting sim")
print(f"parameters: \n viscosity={viscosity}, \n inlet_velocity={inlet_velocity}, \n timestep={timestep}")
with TaskManager():
    while t < tend:
        gfu_prev = gfu.vec.CreateVector()
        gfu_prev.data = gfu.vec

        convect = conv.Apply(gfu_prev)
        
        rhs = m.mat * gfu_prev - timestep * convect
        
        res = rhs - mstar.mat * gfu_bnd.vec
        gfu.vec.data = gfu_bnd.vec + inv * res

        t += timestep
        i += 1

        if i % 10 == 0:
            print(f"Time: {t:.3f} / {tend:.3f}", end='\r')
            Redraw() 
            
            try:
                import netgen.gui
                netgen.gui.ProcessEvents() 
            except AttributeError:
                pass

print("done")

# Plot the final frame using pyplot
import matplotlib.pyplot as plt
import numpy as np

# Get the final velocity field
u_final = gfu.components[0]

# Sample the final field on a regular grid and keep only points inside the mesh
xmin, xmax = 0.0, 2.2
ymin, ymax = 0.0, 0.41
nx, ny = 220, 60

x = np.linspace(xmin, xmax, nx)
y = np.linspace(ymin, ymax, ny)
Xg, Yg = np.meshgrid(x, y)

Ux = np.full_like(Xg, np.nan, dtype=float)
Uy = np.full_like(Yg, np.nan, dtype=float)

for iy in range(ny):
    for ix in range(nx):
        px = float(Xg[iy, ix])
        py = float(Yg[iy, ix])
        try:
            mip = mesh(px, py)
            if mip is None:
                continue
            val = u_final(mip)
            Ux[iy, ix] = val[0]
            Uy[iy, ix] = val[1]
        except Exception:
            # Point outside of curved mesh or not evaluable in current element
            pass

# print reynolds number
average_velocity = np.nanmean(np.sqrt(Ux**2 + Uy**2))
reynolds_number = (average_velocity * 2.2) / viscosity
print(f"Reynolds number: {reynolds_number:.2f}")

# Create figure with subplots
fig, axes = plt.subplots(1, 1, figsize=(10, 4))

# Velocity magnitude
u_mag = np.sqrt(Ux**2 + Uy**2)
pcm = axes.pcolormesh(Xg, Yg, u_mag, shading='auto', cmap='viridis')
axes.set_xlabel('X')
axes.set_ylabel('Y')
axes.set_title('Final Velocity Magnitude')
axes.set_aspect('equal')
plt.colorbar(pcm, ax=axes, label='|v|')

# # Vectors
# step = 4
# Xq = Xg[::step, ::step]
# Yq = Yg[::step, ::step]
# Uq = Ux[::step, ::step]
# Vq = Uy[::step, ::step]
# Mq = np.isfinite(Uq) & np.isfinite(Vq)

# axes[1].quiver(
#     Xq[Mq], Yq[Mq], Uq[Mq], Vq[Mq],
#     np.sqrt(Uq[Mq]**2 + Vq[Mq]**2),
#     cmap='plasma'
# )
# axes[1].set_xlabel('X')
# axes[1].set_ylabel('Y')
# axes[1].set_title('Final Velocity Field')
# axes[1].set_aspect('equal')

plt.tight_layout()
plt.show()


