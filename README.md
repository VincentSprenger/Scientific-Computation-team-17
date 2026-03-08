# Scientific-Computation-team-17


The code is split into 3 parts respective on the people who worked on it, vincents part covers the first part of assignment set 1, andreas covers the second part and barts the third.


# Assignment 2

## Monte Carlo DLA

Requirements: Python 3,numpy, matplotlib

contains a class for running a random walk simulation of DLA. the size of the grid, the sticking probability ps and the seed used for the class can be set when initializing the class. 
The simulation can be ran using class.run(), passing the desired number of walkers as an argument. The results can be plotted with class.plot().  
simulations using a ps value <0.1 tend to take a long time so it is recommended you dont go below that threshold.

## Gray-Scott Reaction Diffusion Equation 

Requirements: Python 3, numpy, matplotlib, numba

Contained in Set2_RD_system.py.
Contains functions for running simulations of the Gray Scott system. 

Quick guide:
run_GS_simulation()
animate()

run_GS_simulation() takes the parameters for the equations, the file path for the returned results, and an initial condition. ("Assignment" or "set+noise").
animate() takes the file path for a results file, and an optional file path to save the video as an mp4.
Example usage is shown at the bottom of the file. Simply run Set2_RD_system.py. 