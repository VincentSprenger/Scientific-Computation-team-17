from numba import stencil,jit
import numpy as np
import matplotlib.pyplot as plt
import time



@stencil
def jacobi_kernel(grid):
    return 0.25 * (grid[0, -1] + grid[0, 1] + grid[-1, 0] + grid[1, 0])

@jit(nopython=True)
def update_grid(grid):
    new_grid = jacobi_kernel(grid)
    set_boundary_conditions(new_grid)
    return new_grid

@jit(nopython=True)
def check_grid(grid, new_grid, threshold=1e-7):
    return np.abs(new_grid - grid).max() < threshold

@jit(nopython=True)
def set_boundary_conditions(grid):
    grid[-1, :] = 0
    grid[:, 0] = 0.5
    grid[:, -1] = 0.5
    grid[0, :] = 1
    # grid[25, 25] = 1

@jit(nopython=True)
def jacobi(size, iterations=False):
    
    grid = np.zeros((size, size))
    new_grid = jacobi_kernel(grid)
    set_boundary_conditions(grid)

    num_iter = 1
    while not check_grid(grid, new_grid):
        grid = new_grid
        new_grid = update_grid(grid)
        num_iter += 1

    if iterations:
        return grid, num_iter
    return grid

@jit(nopython=True)
def gauss_seidel(size, iterations=False):
    grid = np.zeros((size, size))
    set_boundary_conditions(grid)

    max_iterations = 10000
    num_iter = 0

    while num_iter < max_iterations:
        old_grid = grid.copy()
        # 1, n-1 to avoid boundaries
        for i in range(1, size-1):
            for j in range(1, size-1):
                grid[i, j] = 0.25 * (grid[i-1, j] + grid[i+1, j] + grid[i, j-1] + grid[i, j+1])
        
        set_boundary_conditions(grid)
        if check_grid(old_grid, grid):
            break
        num_iter += 1

    if iterations:
        return grid, num_iter
    return grid

@jit(nopython=True)
def successive_over_relaxation(size, omega, threshold=1e-7, iterations=False):
    grid = np.zeros((size, size))
    set_boundary_conditions(grid)

    max_iterations = 10000
    num_iter = 0

    while num_iter < max_iterations:
        old_grid = grid.copy()
        # 1, n-1 to avoid boundaries
        for i in range(1, size-1):
            for j in range(1, size-1):
                grid[i, j] = omega/4 * (grid[i-1, j] + grid[i+1, j] + grid[i, j-1] + grid[i, j+1])+(1-omega)*grid[i, j]
        
        set_boundary_conditions(grid)
        if check_grid(old_grid, grid, threshold=threshold):
            break
        num_iter += 1

    if iterations:
        return grid, num_iter
    return grid

def analyse_trend(size, omega):
    for threshold in [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
        grid = successive_over_relaxation(size, omega,threshold=threshold)
        #plot middle line
        plt.plot(grid[:, 25], label=f'SOR (omega={omega}, threshold={threshold})')
    plt.legend()
    plt.title('Concentration distribution along the middle line for different thresholds')
    plt.xlabel('Position')
    plt.ylabel('Concentration')
    plt.show()

    
if __name__ == "__main__":
    j_grid = jacobi(52)
    plt.imshow(j_grid, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Steady State diffusion')
    plt.show()

    gs_grid = gauss_seidel(52)
    plt.imshow(gs_grid, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Steady State diffusion (Gauss-Seidel)')
    plt.show()

    omega = 1.75
    sor_grid = successive_over_relaxation(52, omega)
    plt.imshow(sor_grid, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title(f'Steady State diffusion (SOR, omega={omega})')
    plt.show()

    # plot middle line
    plt.plot(j_grid[:, 25], label='Jacobi')
    plt.plot(gs_grid[:, 25], label='Gauss-Seidel')
    plt.plot(sor_grid[:, 25], label=f'SOR (omega={omega})')
    plt.legend()
    plt.title('Concentration distribution along the middle line')
    plt.xlabel('Position')
    plt.ylabel('Concentration')
    plt.show()

    # analyse tendency towards convergence
    # analyse_trend(52, omega)