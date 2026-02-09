
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class string_wave:
    def __init__(self,dt, c ):
        self.dt = dt
        self.c = c
        self.N = 100
        self.L = 1
        self.dx = self.L / self.N
        self.x_points = np.linspace(0, self.L, self.N + 1 )
        self.y_pos = np.zeros(len(self.x_points))
        self.y_old = np.zeros(len(self.x_points))
        self.t = 0
        self.wave_history = []  # Store all wave states
        self.time_history = []
        self.first_step = True  # Flag to handle first time step specially

    def string_equation(self):
        y_old = self.y_old.copy()
        y_current = self.y_pos.copy()
        r = (self.c * self.dt / self.dx) ** 2  # Courant number squared
        
        # Calculate new position using finite difference scheme
        y_new = np.zeros(len(self.x_points))
        
        if self.first_step:
            # For first step with zero initial velocity: y_old = y_current
            for i in range(1, len(self.x_points) - 1):
                y_new[i] = (r / 2) * (y_current[i + 1] - 2 * y_current[i] + y_current[i - 1]) + y_current[i]
            self.first_step = False
        else:
            # Standard leapfrog scheme
            for i in range(1, len(self.x_points) - 1):
                y_new[i] = (r * (y_current[i + 1] - 2 * y_current[i] + y_current[i - 1]) 
                           + 2 * y_current[i] - y_old[i])
        
        # Boundary conditions
        y_new[0] = 0
        y_new[-1] = 0
        
        # Update for next iteration
        self.y_old = y_current
        self.y_pos = y_new
        self.t += self.dt


    def run(self, t_max, start_con):
          # Store all time steps
        
        if start_con == 1:
            for i in range(len(self.x_points)):
                self.y_pos[i] = np.sin(2 * np.pi * (self.x_points[i]))

        if start_con == 2:
            for i in range(len(self.x_points)):
                self.y_pos[i] = np.sin(5 * np.pi * (self.x_points[i]))
        if start_con == 3:
            for i in range(len(self.x_points)):
                if self.x_points[i] > 0.25 and self.x_points[i] < 0.5:
                    self.y_pos[i] = np.sin(5 * np.pi * (self.x_points[i]))
                else:
                    self.y_pos[i] = 0

        # Store initial condition
        self.wave_history.append(self.y_pos.copy())
        self.time_history.append(self.t)

        while self.t < t_max:
            self.string_equation()
            self.wave_history.append(self.y_pos.copy())
            self.time_history.append(self.t)

    def animate_wave(self):
        """Create and display animation of the wave movement"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Set up the plot limits
        ax.set_xlim(0, self.L)
        ax.set_ylim(-1.5, 1.5)  # Fixed y-limits for consistent visualization
                    #max([max(wave) for wave in self.wave_history]) + 0.5)
        ax.set_xlabel('Position (x)')
        ax.set_ylabel('Displacement (y)')
        ax.set_title('Wave Movement on String')
        ax.grid(True, alpha=0.3)
        
        # Initialize line object
        line, = ax.plot([], [], lw=2, color='blue')
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
        
        def init():
            line.set_data([], [])
            time_text.set_text('')
            return line, time_text
        
        def animate(frame):
            line.set_data(self.x_points, self.wave_history[frame])
            time_text.set_text(f'Time: {self.time_history[frame]:.3f}s')
            return line, time_text
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                      frames=len(self.wave_history),
                                      interval=1, blit=True, repeat=True)
        
        plt.tight_layout()
        plt.show()


string1 = string_wave(0.001, 1)
string1.run(2, 3)
string1.animate_wave()




    

            


