import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
import time
 
#--------------------------------------------------------------------------------------------------------
#Config
Lx, Ly = 10.0, 8.0 #domain size (meters)
dx = 0.05 #grid spacing - lower=slower
c_light= 3e8
frequency = 2.4e9 #2.4GHz
k_scale = 3 #scale factor
k0 = 2 * np.pi * frequency / c_light / k_scale
 
wt = 0.15 #wall thickness
n_air = 1.0 + 0j
n_wall = 2.5 + 0.5j
 
A_amp = 1e4  #Gaussian source amplitude
sigma = 0.2  # Gaussian source width
excl_area = 0.5 #exclude area around measurement point
measure_rad = 0.05 # measurement radius 
chunk_size = 15000 #used to speed up solving. higher = more ram usage - might want to lower it

mpts = [(1, 5, 'Living room'), (2, 1, 'Kitchen'), (9, 1, 'Bathroom'), (9, 7, 'Bedroom'),]
#--------------------------------------------------------------------------------------------------------

#GRID SETUP
x = np.linspace(0, Lx, round(Lx/dx) + 1)
y = np.linspace(0, Ly, round(Ly/dx) + 1)
Nx, Ny = len(x), len(y)
X, Y = np.meshgrid(x, y, indexing='ij') 
 
print(f"Grid: {Nx} × {Ny} = {Nx*Ny:,} points   dx = {dx} m   k0 = {k0:.2f} rad/m")
 
# refractive index map
n_map = np.full((Nx, Ny), n_air, dtype=complex)

def fill_wall(x0, x1, y0, y1):
    ix = np.where((x >= x0) & (x <= x1))[0]
    jy = np.where((y >= y0) & (y <= y1))[0]
    if len(ix) and len(jy):
        n_map[ix[0]:ix[-1]+1, jy[0]:jy[-1]+1] = n_wall
 
#outer walls
fill_wall(0, wt, 0, Ly)
fill_wall(Lx-wt, Lx, 0, Ly)
fill_wall(0, Lx, 0, wt)
fill_wall(0,Lx, Ly-wt, Ly)
 
#internal walls:
#horizontal walls
fill_wall(0, 3, 3.0-wt/2, 3.0+wt/2) # horizontal left
fill_wall(4, 6, 3.0-wt/2, 3.0+wt/2) # horizontal middle
fill_wall(7, 10, 3.0-wt/2, 3.0+wt/2) # horizontal right

#vertical walls
fill_wall(6.0-wt/2, 6.0+wt/2, 3.0, Ly) # living room
fill_wall(2.5-wt/2, 2.5+wt/2, 0, 2.0 ) # kitchen
fill_wall(7.0-wt/2, 7.0+wt/2, 0, 1.5 ) # bathroom bottom wall
fill_wall(7.0-wt/2, 7.0+wt/2, 2.5, 3.0) # bathroom top wall (small wall above door)
 
k_map = k0 * n_map # complex wavenumber at every grid point
 

# MATRIX PREP
print("building matrix")
t0 = time.time()
 
N = Nx * Ny
rows, cols, vals = [], [], []
 
def add(r, c, v):
    rows.append(np.asarray(r).ravel())
    cols.append(np.asarray(c).ravel())
    vals.append(np.asarray(v, dtype=complex).ravel())
 
# interior points
ii, jj = np.meshgrid(np.arange(1, Nx-1), np.arange(1, Ny-1), indexing='ij')
ii, jj = ii.ravel(), jj.ravel()
r_int  = ii * Ny + jj
 
add(r_int, r_int, k_map[ii,jj]**2 * dx**2 - 4)
add(r_int, (ii+1)*Ny + jj, np.ones(len(r_int)))
add(r_int, (ii-1)*Ny + jj, np.ones(len(r_int)))
add(r_int, ii*Ny + (jj+1), np.ones(len(r_int)))
add(r_int, ii*Ny + (jj-1), np.ones(len(r_int)))

#BOUNDARY CONDITIONS
#left boundary
j = np.arange(Ny)
left_boundary = j 
add(left_boundary, left_boundary, 1 - 1j * k_map[0,  j] * dx)
add(left_boundary, Ny + j, -np.ones(Ny))
 
#right boundary
right_boundary = (Nx-1)*Ny + j
add(right_boundary, right_boundary, 1 - 1j * k_map[Nx-1, j] * dx)
add(right_boundary, (Nx-2)*Ny + j, -np.ones(Ny))
 
#bottom boundary 
i = np.arange(1, Nx-1)
bot_boundary = i * Ny
add(bot_boundary, bot_boundary, 1 - 1j * k_map[i, 0] * dx)
add(bot_boundary, i*Ny + 1, -np.ones(Nx-2))
 
#top boundary
top_boundary = i * Ny + (Ny-1)
add(top_boundary, top_boundary, 1 - 1j * k_map[i, Ny-1] * dx)
add(top_boundary, i*Ny + (Ny-2), -np.ones(Nx-2))
 
rows = np.concatenate(rows).astype(np.int32)
cols = np.concatenate(cols).astype(np.int32)
vals = np.concatenate(vals)
A    = coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
 
print(f"Matrix done ")
 
#LU factorization:
print("LU factorizing")
t0 = time.time()
LU = splu(A)
print(f" LU done in {time.time()-t0:.2f}s")

 
#______________________________________________________________________________________________#
#HELPER FUNCTIONS

wall_clearance = 0.0  #used to try moving router away from walls, does nothing when 0
 
def build_source(xr, yr):
    #Gaussian source vector for router
    f = A_amp * np.exp(-((X-xr)**2 + (Y-yr)**2) / (2*sigma**2))
    return (dx**2 * f).ravel()
 
def solve_field(xr, yr):
    #Return complex wave field u (Nx, Ny).
    return LU.solve(build_source(xr, yr)).reshape(Nx, Ny)
 
#compute measurement masks once, then reuse
m_masks = []
for xm, ym, _ in mpts:
    mask = (X-xm)**2 + (Y-ym)**2 <= measure_rad**2
    m_masks.append(mask.ravel())
 
def measure_signal_single(S):
    #Total score + per-room scores
    #per-room not really that relevant anymore, used in a previous implementation
    room_scores = []
    for xm, ym, _ in mpts:
        mask = (X-xm)**2 + (Y-ym)**2 <= measure_rad**2
        room_scores.append(float(np.mean(S[mask])) if mask.any() else 0.0)
    return sum(room_scores), room_scores
 
def measure_signal_batch(batch_u):

    #Compute total signal scores for a batch of solutions.
    batch_signal = np.abs(batch_u)**2 #batch_u = (N, batch_size)
    scores  = np.zeros(batch_u.shape[1])
    for mask in m_masks:
        # mean over masked grid points for each solution in batch
        pts = batch_signal[mask, :]
        scores += pts.mean(axis=0) if pts.shape[0] > 0 else 0.0
    return scores
 
def is_valid(ix, jy):
    #makes sure we pick a valid cell
    xr, yr = x[ix], y[jy]
 
    #make sure we are not testing wall pints
    if np.real(n_map[ix, jy]) > 1.5:
        return False
 
    #avoid it being too close to the outer wall
    if xr < wall_clearance or xr > Lx - wall_clearance:
        return False
    if yr < wall_clearance or yr > Ly - wall_clearance:
        return False
 
    # keep it .5m away from the measurement points
    for xm, ym, _ in mpts:
        if (xr-xm)**2 + (yr-ym)**2 < excl_area**2:
            return False
 
    return True
#__________________________________________________________________________________________________#
 

# SAMPLING METHOD

# Brute force Scan, optimized via batching.
# Step 1: collect all valid candidate positions first

candidates = [] # list of (ix,jy)
for ix in range(Nx):
    for jy in range(Ny):
        if is_valid(ix, jy):
            candidates.append((ix, jy))
 
n_cands = len(candidates)
print(f"{n_cands} positions will be tested")
 
# Step 2: batch back-substitution
score_map = np.full((Nx, Ny), np.nan)
all_scores = np.zeros(n_cands)

t0 = time.time()
for start in range(0, n_cands, chunk_size):
    batch     = candidates[start : start + chunk_size]
    #build source matrix, then score
    F_chunk  = np.column_stack([build_source(x[ix], y[jy]) for ix, jy in batch])
    U_chunk  = LU.solve(F_chunk)
    sc_chunk  = measure_signal_batch(U_chunk)

    #store results
    for k, (ix, jy) in enumerate(batch):
        score_map[ix, jy] = sc_chunk[k]
        all_scores[start + k] = sc_chunk[k]
 
    #prints progress to check if its too slow
    if (start // chunk_size) % 5 == 0:
        done = min(start + chunk_size, n_cands)
        print(f"    {done}/{n_cands} positions  "
              f"({done/n_cands*100:.1f}%)  "
              f"elapsed: {time.time()-t0:.1f}s")
 
elapsed = time.time() - t0
print(f"  Done — {n_cands} positions in {elapsed:.1f}s  "
      f"({elapsed/n_cands*1000:.2f} ms/solve effective)")
 
#best spot
best_spot = int(np.argmax(all_scores))
best_ix, best_jy = candidates[best_spot]
best_x, best_y   = x[best_ix], y[best_jy]


#-------------------------------------------------------------------
#RESULTS
#-------------------------------------------------------------------
u_best = solve_field(best_x, best_y)
S_best = np.abs(u_best)**2
total, rooms = measure_signal_single(S_best)
 
print("\n" + "=" * 60)
print(f"Best router position : ({best_x:.2f}, {best_y:.2f}) m")
print(f"Total signal score : {total:.6e}")
print()
for (xm, ym, name), sc in zip(mpts, rooms):
    frac = sc / total * 100 if total > 0 else 0
    print(f"  {name:12s}  ({xm},{ym})   score = {sc:.4e}   ({frac:.1f}% of total)")
print("=" * 60)
 
wall_mask = np.real(n_map) > 1.5


# FIG 1 — SIGNAL FIELD
plt.figure(figsize=(7,6))

S_dB = 10 * np.log10(S_best.T + 1e-30)
vmax = S_dB.max()

im = plt.imshow(S_dB, origin='lower', extent=[0,Lx,0,Ly], cmap='jet', aspect='equal', vmin=vmax-40, vmax=vmax)

plt.colorbar(im, label='relative signal strentgh')

# walls
plt.contourf(x, y, wall_mask.T.astype(float), levels=[0.5, 1.5], colors='k', alpha=0.35)

# router
plt.plot(best_x, best_y, 'w*', ms=16, markeredgecolor='black', markeredgewidth=1.5)

# measurement points
for xm, ym, name in mpts:
    plt.plot(xm, ym, 'w^', ms=8)
    plt.text(xm+0.15, ym+0.1, name, color='black', fontsize=7)

plt.title(f'wifi signal coverage at 2.4GHz\nrouter = ({best_x:.2f}, {best_y:.2f}) m, score = {total:.4e}')
plt.xlabel('X (meters)')
plt.ylabel('Y (meters)')

plt.tight_layout()
plt.savefig('signal_field.png', dpi=150)
plt.show()

#  FIG 2 — FLOOR PLAN
plt.figure(figsize=(7,6))
plt.imshow(np.real(n_map).T,origin='lower',extent=[0,Lx,0,Ly],cmap='gray_r',aspect='equal',vmin=1, vmax=3)

# measurement points + exclusion zones
for xm, ym, name in mpts:
    plt.plot(xm, ym, 'b^', ms=8)
    plt.text(xm+0.15, ym+0.1, name, fontsize=7)

    circ = plt.Circle((xm, ym), excl_area, color='blue', fill=False, ls='--', lw=1, alpha=0.5)
    plt.gca().add_patch(circ)

plt.title('Floor plan')
plt.xlabel('X (meters)')
plt.ylabel('Y (meters)')
plt.tight_layout()
plt.savefig('floor_plan.png', dpi=150)
plt.show()