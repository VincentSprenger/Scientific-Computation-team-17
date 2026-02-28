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
        self.candidates_grid = np.zeros((grid_size, grid_size))
        self.eta = eta  
        self.SOR_omega = SOR_omega
        #self.changes_dict = {}


    def test(self):
        print(self.grid)
        self.SOR_step() 
        print(self.candidates_grid)




    def probability(self, x, y):
        c_ij = self.candidates_grid[x, y] ** self.eta
        sum_c_ij = np.sum((self.candidates_grid) ** self.eta)
        return c_ij / sum_c_ij if sum_c_ij > 0 else 0
    
    def SOR_step(self):
        self.candidates_grid.fill(0)  # Reset candidate values
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                #position we are investigating
                if self.grid[x, y]:  # If it's occupied, skip
                    continue
                # Calculate the candidate value based on neighbors
                neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
                for nx, ny in neighbors:
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                        if self.grid[nx, ny]:  # If neighbor is occupied
                            self.candidates_grid[x, y] += self.SOR_omega / 4
       
    def probability_step(self):
        #print(np.nonzero(self.candidates_grid))
        for x, y in zip(*np.nonzero(self.candidates_grid)):
            prob = self.probability(x, y)
            if np.random.rand() < prob:
                self.grid[x, y] = True  # Occupy the site
                self.candidates_grid[x, y] = 0  # Reset candidate value


    def run(self, steps=10):
        for _ in tqdm(range(steps)):
            self.SOR_step()
            self.probability_step()

    def plot_cluster(self):
        flipped = np.flipud(self.grid)
        plt.imshow(flipped, cmap='viridis', origin='lower')
        plt.title('DLA Cluster')
        plt.axis('off')
        plt.show()



DLA_test = DLA(grid_size=100, eta=0.5)
DLA_test.run(steps=2000)
DLA_test.plot_cluster()