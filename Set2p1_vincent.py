import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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
        p_list = p_list / np.sum(p_list)
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

    def plot_cluster(self):
        flipped = np.flipud(self.grid)
        plt.imshow(flipped, cmap='viridis', origin='lower')
        plt.title('DLA Cluster')
        plt.axis('off')
        plt.show()



DLA_test = DLA(grid_size=100, eta=0.1, SOR_omega=1.5)
DLA_test.run(600)
DLA_test.plot_cluster()