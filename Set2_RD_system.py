import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import matplotlib.animation as animation

def empty_grids(size):
    #+2 for boundary
    U = np.zeros((size, size))
    V = np.zeros((size, size))
    return U, V

def add_noise(grid, scale=0.01):
    noise = np.random.normal(0, scale, grid.shape)
    return grid + noise

def initialize_grids(size, type='assignment'):
    size = size+2 # account for boundary
    U,V = empty_grids(size)
    if type == 'assignment':
        U+=0.5
        left_of_center = int(0.5*size-5)
        right_of_center = int(0.5*size+5)
        V[left_of_center:right_of_center, left_of_center:right_of_center] = 0.25
    elif type == 'set+noise':
        U+=0.5
        left_of_center = int(0.5*size-5)
        right_of_center = int(0.5*size+5)
        V[left_of_center:right_of_center, left_of_center:right_of_center] = 0.25
        U = add_noise(U, scale=0.1)
        V = add_noise(V, scale=0.1)
    return U, V

@jit(nopython=True)
def laplacian(grid):
    return (grid[2:, 1:-1] + grid[:-2, 1:-1] + grid[1:-1, 2:] + grid[1:-1, :-2] - 4*grid[1:-1, 1:-1])

@jit(nopython=True)
def forward_euler(U,V,params):
    #unpack params
    f, k, Du, Dv, dt = params

    new_U = np.zeros_like(U)
    new_V = np.zeros_like(V)

    new_U[1:-1, 1:-1] = U[1:-1, 1:-1] + (Du*laplacian(U) - U[1:-1, 1:-1]*V[1:-1, 1:-1]**2 + f*(1-U[1:-1, 1:-1]))*dt
    new_V[1:-1, 1:-1] = V[1:-1, 1:-1] + (Dv*laplacian(V) + U[1:-1, 1:-1]*V[1:-1, 1:-1]**2 - (f+k)*V[1:-1, 1:-1])*dt

    return new_U, new_V

@jit(nopython=True)
def simulate(U, V, steps, params=None):
    history = np.zeros((steps, U.shape[0], U.shape[1], 2))
    if params is None:
        # params= (f, k, Du, Dv, dt)
        params = (0.0050, 0.0610, 0.16, 0.08, 1)
    for i in range(steps):
        U, V = forward_euler(U, V, params)
        history[i, :, :, 0] = U
        history[i, :, :, 1] = V
    return history

def plot(U, V):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title('U concentration')
    plt.imshow(U, cmap='viridis',vmin=0, vmax=1)
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.title('V concentration')
    plt.imshow(V, cmap='viridis',vmin=0, vmax=1)
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()

def animate():
    # only animate every 100 steps to save memory and speed up animation
    history = np.load('simulation_data.npy')
    fig, axes = plt.subplots(1,2, figsize=(12, 5))
    # 0 for U, 1 for V
    grid_num = 1
    im_u = axes[0].imshow(history[0, :, :, 0], cmap='Blues', vmin=0, vmax=1)
    im_v = axes[1].imshow(history[0, :, :, 1], cmap='Reds', vmin=0, vmax=1)

    axes[0].set_title("U Component")
    axes[1].set_title("V Component")

    def animate_frame(i):
        im_u.set_array(history[i, :, :, 0])
        im_v.set_array(history[i, :, :, 1])
        # Return a list of all artists to be redrawn
        return [im_u, im_v]

    frame_indices = np.linspace(0, history.shape[0] - 1, 700, dtype=int)

    ani = animation.FuncAnimation(fig,animate_frame,frames=frame_indices,interval=50,blit=True)
    plt.show()

# print("Initializing grids...")
# U, V = initialize_grids(size=100, type='set+noise')
# print("Starting simulation...")
# params = (0.035, 0.065, 0.16, 0.05, 1)
# history = simulate(U, V, steps=20000, params=params)
# np.save('simulation_data.npy', np.array(history))
# print("Simulation complete. Plotting results...")
# print("Done")

animate()

# special cases
# default
# (0.035, 0.06, 0.16, 0.08, 1)
# cool cell splitting behaviour
# params = (0.035, 0.065, 0.16, 0.05, 1) # f, k, Du, Dv, dt