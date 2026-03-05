import sys
print("RUNNING:", sys.executable)


import numpy as np
import matplotlib.pyplot as plt


class DLA_MC:
    def __init__(self, size, ps=1.0, seed=None):
        self.seed = seed
        self.size = size
        self.ps = ps
        self.grid = np.zeros((size, size), dtype=int)
        self.grid[size - 1, size // 2] = 1

        if seed is not None:
            np.random.seed(seed)

    def step(self, x, y):
        #gets random number from 0-3, each one tied to a direction
        # 0 = up | 1 = down | 2 = left | 3 = right
        direction = np.random.randint(4)

        if direction == 0:
            x -= 1

        elif direction == 1:
            x += 1

        elif direction == 2:
            y = (y - 1) % self.size
    
        elif direction == 3:
            y = (y + 1) % self.size

        # top/bottom boundary check
        if x < 0 or x >= self.size:
            return None

        return x, y
    
    def run(self, walkers):
        stuck = 0
        while stuck < walkers:
            # release walker at top
            x, y = 0, np.random.randint(0, self.size)
            
            while True:
                result = self.step(x, y)
                if result is None:
                    break
                x, y = result

                # check if next to cluster
                neighbors = []
                if x > 0: neighbors.append((x-1, y))
                if x < self.size - 1: neighbors.append((x+1, y))
                neighbors.append((x,(y-1) % self.size))
                neighbors.append((x,(y+1) % self.size))

                if any(self.grid[i, j] == 1 for i, j in neighbors):
                    if np.random.random() <= self.ps:
                        self.grid[x, y] = 1
                        stuck += 1
                        break

    def plot(self):
        plt.figure(figsize=(6, 6))
        plt.imshow(self.grid, cmap="binary", origin="upper")
        plt.title(f"Monte Carlo DLA  |  ps = {self.ps}")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

#PS = 1 acts the same as no ps, so it should work for part C
simulation = DLA_MC(size=100, ps=1.0)
simulation.run(walkers=600)
simulation.plot()
#TODO: add test for the other ps values