import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from timeit import default_timer as timer

class DLA:
    def __init__(self, grid_size=5, eta =1, SOR_omega = 1, num_particles = 1):
        self.grid_size = grid_size
        self.num_particles = num_particles
        self.grid = np.zeros((grid_size, grid_size), dtype=bool)
        self.center = grid_size // 2
        self.grid[self.grid_size - 1, self.center] = True  # Seed particle at the center
        self.concentration_grid = np.zeros((grid_size, grid_size))
        self.candidates_grid = np.zeros((grid_size, grid_size), dtype=bool)
        self.eta = eta  
        self.SOR_omega = SOR_omega
        #self.changes_dict = {}


    def test(self):
        print(self.grid)
        self.SOR_step() 
        self.SOR_step() 
        self.SOR_step() 
        self.SOR_step() 
        self.SOR_step() 
        self.SOR_step() 
        self.SOR_step() 
        self.SOR_step() 
        self.SOR_step() 
        self.candidates_step()
        print(self.candidates_grid)
        self.probability_step()
        print(self.grid)
        self.SOR_step()
        self.SOR_step()
        self.SOR_step()
        #rint(self.concentration_grid)
        




    def probability(self, x, y):
        c_ij = self.concentration_grid[x, y] ** self.eta
        sum_c_ij = np.sum((self.concentration_grid** self.eta) )
        return c_ij / sum_c_ij if sum_c_ij > 0 else 0
    
    def candidates_step(self):
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.grid[x, y]:  # If occupied
                    # Mark neighbors as candidates (only within bounds)
                    if x - 1 >= 0:
                        self.candidates_grid[x-1, y] = True
                    if x + 1 < self.grid_size:
                        self.candidates_grid[x+1, y] = True
                    if y - 1 >= 0:
                        self.candidates_grid[x, y-1] = True
                    if y + 1 < self.grid_size:
                        self.candidates_grid[x, y+1] = True
    
    def candidates_step_parallel(self):
        """Vectorized/parallelized candidates step using NumPy operations"""
        # Start fresh
        self.candidates_grid.fill(False)
        
        # Find all occupied cells
        occupied_x, occupied_y = np.where(self.grid)
        
        # Mark all neighbors of occupied cells as candidates
        for x, y in zip(occupied_x, occupied_y):
            # Mark x-1 neighbor
            if x - 1 >= 0:
                self.candidates_grid[x-1, y] = True
            # Mark x+1 neighbor
            if x + 1 < self.grid_size:
                self.candidates_grid[x+1, y] = True
            # Mark y-1 neighbor
            if y - 1 >= 0:
                self.candidates_grid[x, y-1] = True
            # Mark y+1 neighbor
            if y + 1 < self.grid_size:
                self.candidates_grid[x, y+1] = True
    
    def SOR_step(self):
        self.concentration_grid[0, :] = 1
        #self.concentration_grid[:, 0] = 1
        #self.concentration_grid[:, -1] = 1
        #print(self.concentration_grid)
        for x in range(1, self.grid_size):
            for y in range(self.grid_size):
                if self.grid[x, y]:  # If occupied, set concentration to 0
                    self.concentration_grid[x, y] = 0
                else:
                    # Average of the four neighbors (only within bounds)
                    neighbors_count = 0
                    neighbors_sum = 0
                    if x - 1 >= 0:
                        neighbors_sum += self.concentration_grid[x-1, y]
                        neighbors_count += 1
                    if x + 1 < self.grid_size:
                        neighbors_sum += self.concentration_grid[x+1, y]
                        neighbors_count += 1
                    if y - 1 >= 0:
                        neighbors_sum += self.concentration_grid[x, y-1]
                        neighbors_count += 1
                    if y + 1 < self.grid_size:
                        neighbors_sum += self.concentration_grid[x, y+1]
                        neighbors_count += 1
                    new_value = neighbors_sum / 4 #neighbors_count if neighbors_count > 0 else 0
                    # SOR update
                    self.concentration_grid[x, y] = (1 - self.SOR_omega) * self.concentration_grid[x, y] + self.SOR_omega * new_value
                    # Clamp to non-negative values
                    self.concentration_grid[x, y] = max(0, self.concentration_grid[x, y])
        #print(self.concentration_grid)
       
        #print(self.candidates_grid)

    def SOR_step_parallel(self):
        """Vectorized/parallelized SOR step using NumPy operations"""
        self.concentration_grid[0, :] = 1
        
        # Create output grid
        new_grid = self.concentration_grid.copy()
        
        # Process each row with vectorized operations
        for x in range(1, self.grid_size):
            # Compute sum of all neighbors for this row (vectorized)
            neighbor_sum = np.zeros(self.grid_size, dtype=float)
            
            # X neighbors
            neighbor_sum += self.concentration_grid[x - 1, :]  # x-1 (always valid)
            if x + 1 < self.grid_size:
                neighbor_sum[0:self.grid_size] += self.concentration_grid[x + 1, :]  # x+1
            
            # Y neighbors (with proper boundary handling - no double counting)
            temp_y = self.concentration_grid[x, :].copy()
            neighbor_sum += np.pad(temp_y[:-1], (1, 0), mode='constant')    # y-1 shift
            neighbor_sum += np.pad(temp_y[1:], (0, 1), mode='constant')     # y+1 shift
            
            new_val = neighbor_sum / 4
            
            # Only update non-occupied cells
            non_occupied_mask = ~self.grid[x, :]
            new_grid[x, non_occupied_mask] = np.maximum(
                0,
                (1 - self.SOR_omega) * self.concentration_grid[x, non_occupied_mask] + 
                self.SOR_omega * new_val[non_occupied_mask]
            )
        
        self.concentration_grid = new_grid

    def probability_step(self):
        #print(np.nonzero(self.candidates_grid))
        x_y_list = []
        p_list = []
        for x, y in zip(*np.nonzero(self.candidates_grid)):
            prob = self.probability(x, y)
            if prob < 0:
                print(self.concentration_grid)
            #print(prob)
            x_y_list.append((x, y))
            p_list.append(prob)
        
        p_list = np.array(p_list, dtype=float)
        p_sum = np.sum(p_list)
        
        # Handle case where all probabilities are zero or sum is NaN
        if p_sum <= 0 or np.isnan(p_sum):
            # Uniform probability distribution as fallback
            p_list = np.ones_like(p_list) / len(p_list)
        else:
            p_list = p_list / p_sum
        
        # Remove any remaining NaNs with uniform distribution
        if np.any(np.isnan(p_list)):
            p_list = np.ones_like(p_list) / len(p_list)
        
        #print(p_list)  # Normalize probabilities
        chosen_node = np.random.choice(len(x_y_list), p=p_list)
        chosen_x, chosen_y = x_y_list[chosen_node]
        self.grid[chosen_x, chosen_y] = True  # Occupy the chosen node
        self.candidates_grid[chosen_x, chosen_y] = 0  # Reset candidate value

    def run(self, steps=10, s_steps=10, r_steps=3):
        start = True
        for _ in tqdm(range(steps)):
            if start:
                for j in range(s_steps):
                    self.SOR_step()
                    
                start = False
            else:
                self.candidates_step()
                self.probability_step()
                #print("probability step done")
                for k in range(r_steps):
                    self.SOR_step()
        
    def run_parallel(self, steps=10, s_steps=10, r_steps=3):
        """Parallelized version using vectorized SOR and candidates steps"""
        start = True
        for _ in tqdm(range(steps)):
            if start:
                for j in range(s_steps):
                    self.SOR_step_parallel()
                    
                start = False
            else:
                self.candidates_step_parallel()
                self.probability_step()
                for k in range(r_steps):
                    self.SOR_step_parallel()

    def plot_cluster(self):
        flipped = np.flipud(self.grid)
        plt.imshow(flipped, cmap='viridis', origin='lower')
        plt.title('DLA Cluster')
        plt.axis('off')
        plt.show()

    def run_test(self, steps):
        start = timer()
        self.run(steps)
        print(f"Execution time: {timer() - start:.4f} seconds")
        self.plot_cluster()

        self.__init__(grid_size=self.grid_size, eta=self.eta, SOR_omega=self.SOR_omega)  # Reset the grid and concentration

        start = timer()
        self.run_parallel(steps)
        print(f"Parallel Execution time: {timer() - start:.4f} seconds")
        self.plot_cluster()


DLA_test = DLA(grid_size=100, eta=1, SOR_omega=1.5)
#DLA_test.run(600)
#DLA_test.plot_cluster()

DLA_test.run_test(500)
#DLA_test.run_parallel(800)
#DLA_test.plot_cluster()