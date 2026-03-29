# Scientific-Computation-team-17


The code is split into 3 parts respective on the people who worked on it, vincents part covers the first part of assignment set 1, andreas covers the second part and barts the third.


# Assignment 2

## Diffusion Limited Aggregation

Requirements: Python 3,numpy, matplotlib, numba, tqdm, timeit

Contains a class called DLA which creates the DLA model, the functions used to run the model are:

DLA.run(steps, s_steps, r_steps) runs the model in its original state. Steps defines how many growth steps the model makes before stopping, s_steps defines how many times the SOR equation is run to initialize the model, r_steps is how many times the SOR equation is run between steps these do not need to be changed.

DLA.plot_cluster() plots the created clusters.

DLA.run_parallel(steps, s_steps, r_steps) works the same as .run() however due to compilation is significantly faster and suggested to use

DLA.run_test(steps) runs the model both as .run() and .run_parallel() with the same params to compare computational time.

just uncomment the functions at the bottom of the code to play with it.

## Monte Carlo DLA

Requirements: Python 3,numpy, matplotlib

can be ran from set2.2_andreas.py

contains a class for running a random walk simulation of DLA. the size of the grid, the sticking probability ps and the seed used for the class can be set when initializing the class. 
The simulation can be ran using class.run(), passing the desired number of walkers as an argument. The results can be plotted with class.plot().  
simulations using a ps value <0.1 tend to take a long time so it is recommended you dont go below that threshold.

## Gray-Scott Reaction Diffusion Equation 

Requirements: Python 3, numpy, matplotlib, numba

Contained in Set2_RD_system.py.
Contains functions for running simulations of the Gray Scott system. 

Quick guide:
```
run_GS_simulation()
animate()
```

run_GS_simulation() takes the parameters for the equations, the file path for the returned results, and an initial condition. ("Assignment" or "set+noise").

animate() takes the file path for a results file, and an optional file path to save the video as an mp4.

Example usage is shown at the bottom of the file. Simply run Set2_RD_system.py. 

<details>
  <summary>Videos</summary>
  f=0.035, k=0.06, du=0.16, dv=0.08, dt=1

  ![Video](./Assignment2/sim_results/DR_default.gif)

  f=0.055, k=0.062, du=0.16, dv=0.08, dt=1
  ![Video](./Assignment2/sim_results/DR_maze_angled.gif)

  f=0.035, k=0.065, du=0.16, dv=0.08, dt=1
  ![Video](./Assignment2/sim_results/DR_mitosis_like.gif)


</details>


# Assignment 3

## WiFi optimization

Requirements: numpy, scipy, matplotlib  
Run directly with: python set3-wifi.py  

finds the optimal WiFi router position in a 2D floor plan by solving the Helmholtz equation. The floor plan is discretized onto a uniform grid, with the 2 materials - air and wall- having a different refractive index. A brute forced sampling approach is used to find the optimal router position
Program outputs 2 pngs, one of the wave propagation pattern and one of the layout of the house.  

All parameters are located at the top of the file under `#Config` and can be changed:

- `dx` — grid spacing in meters, lower is more accurate but slower
- `k_scale` — wavenumber scaling factor
- `frequency` — WiFi frequency in Hz
- `wt` — wall thickness in meters
- `n_air` — refractive index of air
- `n_wall` — refractive index of walls
- `A_amp` — Gaussian source amplitude
- `sigma` — Gaussian source width in meters
- `excl_area` — exclusion radius around measurement points in meters
- `measure_rad` — averaging radius for signal measurement in meters
- `chunk_size` — batch size for solving, higher is faster but uses more RAM

