from numba import stencil,jit
import numpy as np
import matplotlib.pyplot as plt
import time

@jit(nopython=True)
def check_grid(grid, new_grid, threshold=1e-7):
    return np.abs(new_grid - grid).max()


@jit(nopython=True)
def top_to_bottom_edge(grid):
    grid[-1, :] = 0
    grid[:, 0] = grid[:, 1]
    grid[:, -1] = grid[:, -2]
    grid[0, :] = 1

@jit(nopython=True)
def sink_edge(grid):
    grid[-1, :] = 0
    grid[:, 0] = 0
    grid[:, -1] = 0
    grid[0, :] = 0

@jit(nopython=True)
def set_boundary_conditions(grid, objects, edge_type=top_to_bottom_edge): 
    """
    Set the boundary conditions for the grid. The object is a nx3 array where each row is (x,y,value)
    """

    for i in range(0, objects.shape[0]):
        x, y, value = objects[i,:]
        grid[x, y] = value

    edge_type(grid)


@jit(nopython=True)
def jacobi(size, objects, threshold=1e-6,max_iterations=5000, edge_type=top_to_bottom_edge):
    size = size+2 # add 2 for boundaries
    grid = np.zeros((size, size))
    new_grid = np.zeros((size, size))
    set_boundary_conditions(grid, objects, edge_type=edge_type)
    max_iterations = 10000
    num_iter = 0
    deltas = np.zeros(max_iterations, dtype=np.float64)

    while num_iter < max_iterations:
        old_grid = grid.copy()
        # 1, n-1 to avoid boundaries
        for i in range(1, size-1):
            for j in range(1, size-1):
                if check_in_objects(i, j, objects):
                    continue
                new_grid[i, j] = 0.25 * (grid[i-1, j] + grid[i+1, j] + grid[i, j-1] + grid[i, j+1])
        
        set_boundary_conditions(new_grid, objects, edge_type=edge_type)
        delta = check_grid(old_grid, new_grid, threshold=threshold)
        deltas[num_iter] = delta
        if delta < threshold:
            break
        grid = new_grid.copy()
        num_iter += 1
    return grid, num_iter, deltas

@jit(nopython=True)
def gauss_seidel(size, objects, threshold=1e-6,max_iterations=5000, edge_type=top_to_bottom_edge):
    size = size+2 # add 2 for boundaries
    grid = np.zeros((size, size))
    set_boundary_conditions(grid, objects, edge_type=edge_type)

    max_iterations = 10000
    num_iter = 0
    deltas = np.zeros(max_iterations, dtype=np.float64)

    while num_iter < max_iterations:
        old_grid = grid.copy()
        # 1, n-1 to avoid boundaries
        for i in range(1, size-1):
            for j in range(1, size-1):
                if check_in_objects(i, j, objects):
                    continue
                grid[i, j] = 0.25 * (grid[i-1, j] + grid[i+1, j] + grid[i, j-1] + grid[i, j+1])
        
        set_boundary_conditions(grid, objects, edge_type=edge_type)
        delta = check_grid(old_grid, grid, threshold=threshold)
        deltas[num_iter] = delta
        if delta < threshold:
            break
        num_iter += 1

    
    return grid, num_iter, deltas

@jit(nopython=True)
def successive_over_relaxation(size, omega, objects, threshold=1e-6,max_iterations=5000, edge_type=top_to_bottom_edge):
    size = size+2 # add 2 for boundaries
    grid = np.zeros((size, size))
    set_boundary_conditions(grid, objects, edge_type=edge_type)

    max_iterations = 10000
    num_iter = 0
    deltas = np.zeros(max_iterations, dtype=np.float64)

    while num_iter < max_iterations:
        old_grid = grid.copy()
        # 1, n-1 to avoid boundaries
        for i in range(1, size-1):
            for j in range(1, size-1):
                if check_in_objects(i, j, objects):
                    continue
                grid[i, j] = omega/4 * (grid[i-1, j] + grid[i+1, j] + grid[i, j-1] + grid[i, j+1])+(1-omega)*grid[i, j]
        
        set_boundary_conditions(grid, objects, edge_type=edge_type)
        delta = check_grid(old_grid, grid, threshold=threshold)
        deltas[num_iter] = delta
        if delta < threshold:
            break
        num_iter += 1

    return grid, num_iter, deltas

def analyse_trend(size, omega):
    for threshold in [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
        grid,iter, delta_list = successive_over_relaxation(size, omega, empty_object(),threshold=threshold)
        #plot middle line
        plt.plot(grid[:, 25], label=f'SOR (omega={omega}, threshold={threshold}, iterations={iter})')
    plt.legend()
    plt.title('Concentration distribution along the middle line for different thresholds')
    plt.xlabel('Position')
    plt.ylabel('Concentration')
    plt.show()

@jit(nopython=True)
def empty_object():
    return np.empty((0, 3), dtype=np.int16)

@jit(nopython=True)
def circle_object(x_center, y_center, radius, value):
    x_center +=1
    y_center +=1

    count = 0
    min_x = x_center - radius
    max_x = x_center + radius
    min_y = y_center - radius
    max_y = y_center + radius

    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            if (x - x_center) ** 2 + (y - y_center) ** 2 <= radius ** 2:
                count += 1
    
    object = np.zeros((count, 3), dtype=np.int16)

    i = 0
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            if (x - x_center) ** 2 + (y - y_center) ** 2 <= radius ** 2:
                object[i, 0] = x
                object[i, 1] = y
                object[i, 2] = value
                i += 1
    return object

@jit(nopython=True)
def square_object(x_start, x_end, y_start, y_end, value):
    # shift 1 to avoid boundaries
    width = x_end - x_start+1
    height = y_end - y_start+1
    num_points = width * height
    
    object = np.zeros((num_points, 3), dtype=np.int16)
    
    count = 0
    for x in range(x_start+1, x_end + 1):
        for y in range(y_start+1, y_end + 1):
            object[count, 0] = x
            object[count, 1] = y
            object[count, 2] = value
            count += 1
            
    return object

@jit(nopython=True)
def check_in_objects(x, y, objects):
    for i in range(objects.shape[0]):
        if objects[i, 0] == x and objects[i, 1] == y:
            return True
    return False

# find optimal omega with golden ratio search
@jit(nopython=True)
def golden_ratio_search(left_bound, right_bound, threshold=1e-3, size = 50):
    size = size+2
    inv_phi = 0.618033

    left_o = right_bound - inv_phi * (right_bound - left_bound)
    right_o = left_bound + inv_phi * (right_bound - left_bound)

    _, right_iter, _ = successive_over_relaxation(size, right_o, empty_object())
    _, left_iter, _ = successive_over_relaxation(size, left_o, empty_object())

    steps = 0
    while abs(right_o - left_o) > threshold and steps < 100:
        
        if right_iter < left_iter:
            left_bound = left_o
            right_bound = right_bound
            left_o = right_o
            left_iter = right_iter
            right_o = left_bound + inv_phi * (right_bound - left_bound)
            _, right_iter, _ = successive_over_relaxation(size, right_o, empty_object())
        else:
            left_bound = left_bound
            right_bound = right_o
            right_o = left_o
            right_iter = left_iter
            left_o = right_bound - inv_phi * (right_bound - left_bound)
            _, left_iter, _ = successive_over_relaxation(size, left_o, empty_object())
        steps += 1
    return (left_o + right_o) / 2

    
if __name__ == "__main__":
    # set no objects
    objects = empty_object()
    j_grid,j_iter, j_deltas = jacobi(50, objects)
    plt.imshow(j_grid, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title(f'Steady State diffusion (Jacobi, {j_iter} iterations)')
    plt.show()

    gs_grid,gs_iter, gs_deltas = gauss_seidel(50, objects)
    plt.imshow(gs_grid, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title(f'Steady State diffusion (Gauss-Seidel, {gs_iter} iterations)')
    plt.show()

    omega = 1.75
    sor_grid,sor_iter, sor_deltas = successive_over_relaxation(50, omega, objects)
    plt.imshow(sor_grid, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title(f'Steady State diffusion (SOR, omega={omega}, {sor_iter} iterations)')
    plt.show()

    omega = 1.1
    sor_grid1,sor_iter1, sor_deltas1 = successive_over_relaxation(50, omega, objects)

    omega = 1.5
    sor_grid2,sor_iter2, sor_deltas2 = successive_over_relaxation(50, omega, objects)

    omega = 1.7
    sor_grid3,sor_iter3, sor_deltas3 = successive_over_relaxation(50, omega, objects)

    omega = 1.9
    sor_grid4,sor_iter4, sor_deltas4 = successive_over_relaxation(50, omega, objects)

    omega = 2
    sor_grid5,sor_iter5, sor_deltas5 = successive_over_relaxation(50, omega, objects)

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
    analyse_trend(50, omega)

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

    # Object playground
    obj = square_object(10, 20, 10, 20, 1)
    obj2 = circle_object(30, 30, 5, 0)
    obj3 = circle_object(15, 35, 3, 1)
    total_obj = np.concatenate((obj, obj2, obj3), axis=0)
    sor_grid_obj, sor_iter_obj, sor_deltas_obj = successive_over_relaxation(50, 1.94, total_obj,edge_type=sink_edge)
    plt.imshow(sor_grid_obj, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title(f'Steady State diffusion with object (SOR, omega=1.94, {sor_iter_obj} iterations)')
    plt.show()