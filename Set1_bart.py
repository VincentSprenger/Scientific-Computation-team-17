from numba import stencil,jit
import numpy as np
import matplotlib.pyplot as plt
import time





@jit(nopython=True)
def check_grid(grid, new_grid, threshold=1e-7):
    return np.abs(new_grid - grid).max()

@jit(nopython=True)
def set_boundary_conditions(grid):
    grid[-1, :] = 0
    grid[:, 0] = grid[:, 1]
    grid[:, -1] = grid[:, -2]
    grid[0, :] = 1
    # grid[25, 25] = 1

@jit(nopython=True)
def jacobi(size, threshold=1e-6,max_iterations=5000):
    
    grid = np.zeros((size, size))
    new_grid = np.zeros((size, size))
    set_boundary_conditions(grid)
    max_iterations = 10000
    num_iter = 0
    deltas = np.zeros(max_iterations, dtype=np.float64)

    while num_iter < max_iterations:
        old_grid = grid.copy()
        # 1, n-1 to avoid boundaries
        for i in range(1, size-1):
            for j in range(1, size-1):
                new_grid[i, j] = 0.25 * (grid[i-1, j] + grid[i+1, j] + grid[i, j-1] + grid[i, j+1])
        
        set_boundary_conditions(new_grid)
        delta = check_grid(old_grid, new_grid, threshold=threshold)
        deltas[num_iter] = delta
        if delta < threshold:
            break
        grid = new_grid.copy()
        num_iter += 1
    return grid, num_iter, deltas

@jit(nopython=True)
def gauss_seidel(size, threshold=1e-6,max_iterations=5000):
    grid = np.zeros((size, size))
    set_boundary_conditions(grid)

    max_iterations = 10000
    num_iter = 0
    deltas = np.zeros(max_iterations, dtype=np.float64)

    while num_iter < max_iterations:
        old_grid = grid.copy()
        # 1, n-1 to avoid boundaries
        for i in range(1, size-1):
            for j in range(1, size-1):
                grid[i, j] = 0.25 * (grid[i-1, j] + grid[i+1, j] + grid[i, j-1] + grid[i, j+1])
        
        set_boundary_conditions(grid)
        delta = check_grid(old_grid, grid, threshold=threshold)
        deltas[num_iter] = delta
        if delta < threshold:
            break
        num_iter += 1

    
    return grid, num_iter, deltas

@jit(nopython=True)
def successive_over_relaxation(size, omega, threshold=1e-6,max_iterations=5000):
    grid = np.zeros((size, size))
    set_boundary_conditions(grid)

    max_iterations = 10000
    num_iter = 0
    deltas = np.zeros(max_iterations, dtype=np.float64)

    while num_iter < max_iterations:
        old_grid = grid.copy()
        # 1, n-1 to avoid boundaries
        for i in range(1, size-1):
            for j in range(1, size-1):
                grid[i, j] = omega/4 * (grid[i-1, j] + grid[i+1, j] + grid[i, j-1] + grid[i, j+1])+(1-omega)*grid[i, j]
        
        set_boundary_conditions(grid)
        delta = check_grid(old_grid, grid, threshold=threshold)
        deltas[num_iter] = delta
        if delta < threshold:
            break
        num_iter += 1

    return grid, num_iter, deltas

def analyse_trend(size, omega):
    for threshold in [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
        grid,iter, delta_list = successive_over_relaxation(size, omega,threshold=threshold)
        #plot middle line
        plt.plot(grid[:, 25], label=f'SOR (omega={omega}, threshold={threshold}, iterations={iter})')
    plt.legend()
    plt.title('Concentration distribution along the middle line for different thresholds')
    plt.xlabel('Position')
    plt.ylabel('Concentration')
    plt.show()

# find optimal omega with golden ratio search
@jit(nopython=True)
def golden_ratio_search(left_bound, right_bound, threshold=1e-3, size = 50):
    size = size+2
    inv_phi = 0.618033

    left_o = right_bound - inv_phi * (right_bound - left_bound)
    right_o = left_bound + inv_phi * (right_bound - left_bound)

    _, right_iter, _ = successive_over_relaxation(size, right_o)
    _, left_iter, _ = successive_over_relaxation(size, left_o)

    steps = 0
    while abs(right_o - left_o) > threshold and steps < 100:
        
        if right_iter < left_iter:
            left_bound = left_o
            right_bound = right_bound
            left_o = right_o
            left_iter = right_iter
            right_o = left_bound + inv_phi * (right_bound - left_bound)
            _, right_iter, _ = successive_over_relaxation(size, right_o)
        else:
            left_bound = left_bound
            right_bound = right_o
            right_o = left_o
            right_iter = left_iter
            left_o = right_bound - inv_phi * (right_bound - left_bound)
            _, left_iter, _ = successive_over_relaxation(size, left_o)
        steps += 1
    return (left_o + right_o) / 2

    
if __name__ == "__main__":
    j_grid,j_iter, j_deltas = jacobi(52)
    plt.imshow(j_grid, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title(f'Steady State diffusion (Jacobi, {j_iter} iterations)')
    plt.show()

    gs_grid,gs_iter, gs_deltas = gauss_seidel(52)
    plt.imshow(gs_grid, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title(f'Steady State diffusion (Gauss-Seidel, {gs_iter} iterations)')
    plt.show()

    omega = 1.75
    sor_grid,sor_iter, sor_deltas = successive_over_relaxation(52, omega)
    plt.imshow(sor_grid, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title(f'Steady State diffusion (SOR, omega={omega}, {sor_iter} iterations)')
    plt.show()

    omega = 1.1
    sor_grid1,sor_iter1, sor_deltas1 = successive_over_relaxation(52, omega)

    omega = 1.5
    sor_grid2,sor_iter2, sor_deltas2 = successive_over_relaxation(52, omega)

    omega = 1.7
    sor_grid3,sor_iter3, sor_deltas3 = successive_over_relaxation(52, omega)

    omega = 1.9
    sor_grid4,sor_iter4, sor_deltas4 = successive_over_relaxation(52, omega)

    omega = 2
    sor_grid5,sor_iter5, sor_deltas5 = successive_over_relaxation(52, omega)

    # plot middle line
    plt.plot(j_grid[:, 25], label=f'Jacobi ({j_iter} iters)')
    plt.plot(gs_grid[:, 25], label=f'Gauss-Seidel ({gs_iter} iters)')
    plt.plot(sor_grid[:, 25], label=rf'SOR ($\omega$={omega}, {sor_iter} iters)')
    plt.legend()
    plt.title('Concentration distribution along the middle line')
    plt.xlabel('Position')
    plt.ylabel('Concentration')
    plt.show()

    # analyse tendency towards convergence
    analyse_trend(52, omega)

    # Convergence

    j_deltas = j_deltas[:j_iter]
    gs_deltas = gs_deltas[:gs_iter]
    sor_deltas1 = sor_deltas1[:sor_iter1]
    sor_deltas2 = sor_deltas2[:sor_iter2]
    sor_deltas3 = sor_deltas3[:sor_iter3]
    sor_deltas4 = sor_deltas4[:sor_iter4]
    sor_deltas5 = sor_deltas5[:sor_iter5]

    plt.plot(j_deltas, label=f'Jacobi ({j_iter} iters)')
    plt.plot(gs_deltas, label=f'Gauss-Seidel ({gs_iter} iters)')
    plt.plot(sor_deltas1, label=fr'SOR ($\omega=1.0$, {sor_iter1} iters)')
    plt.plot(sor_deltas2, label=fr'SOR ($\omega=1.5$, {sor_iter2} iters)')
    plt.plot(sor_deltas3, label=fr'SOR ($\omega=1.7$, {sor_iter3} iters)')
    plt.plot(sor_deltas4, label=fr'SOR ($\omega=1.9$, {sor_iter4} iters)')
    plt.plot(sor_deltas5, label=fr'SOR ($\omega=2.0$, {sor_iter5} iters)')
    
    plt.legend()
    plt.title('Error over iterations')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.show()

    
    # plot optimal omega over size
    omegas = []
    sizes = np.linspace(10, 100, 11, dtype=int)
    for size in sizes:
        optimal_omega = golden_ratio_search(1.0, 1.999, threshold=1e-3, size=size)
        omegas.append(optimal_omega)
        print(f'Optimal omega for size {size}: {optimal_omega}')
    plt.plot(sizes, omegas, label='Optimal omega over size')
    plt.xlabel('Size')
    plt.ylabel('Optimal omega')
    plt.title('Optimal omega over size')
    plt.legend()
    plt.show()