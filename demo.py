"""
This demo shows workflow and limits of solving PDES with the
Euler scheme. Try larger time steps with implicit and explicit
solution schemes.
"""
import matplotlib.pyplot as plt
import numpy as np
import pde1d as pde

# Switch between implicit and explicit
explicit = True

# Init space domain
dx = 1
x = np.arange(0, 100, dx)

# Init time domain
dt = 1                          # try 5 and 10 seconds (unstable)
start_time = np.datetime64("2021-01-01T00:00:00")
stop_time = np.datetime64("2021-01-01T00:30:00")
time = np.arange(start_time, stop_time, np.timedelta64(dt, 's'))

# Define initial conditions
u0 = np.zeros_like(x)
u0[40:60] = 1

# This will be a diffusion problem, define the diffusion constant
k = 0.1 * np.ones_like(x)

# Define system of equations
if explicit:
    coefs = pde.coefs_diffusion(x, time, k, implicit=False)
else:
    coefs = pde.coefs_diffusion(x, time, k, implicit=True)

# Define the boundary conditions as zero flux
boundary_0 = 0, dx
boundary_j = 0, dx

# Instanciate the problem to solve
if explicit:
    system = pde.ExplicitSystem(coefs, boundary_0, boundary_j)
else:
    system = pde.ImplicitSystem(coefs, boundary_0, boundary_j)

# Time step the solution using the Euler explicit algorithm
solution = pde.time_step_euler(u0, time, system)

# Visualize
pde.plot_solution(x, time, solution)
plt.show()
