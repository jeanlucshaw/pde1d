"""
"""
import matplotlib.pyplot as plt
import mxtoolbox.plot as pt
import numpy as np
import pde1d as pde

# Switch between implicit and explicit
explicit = False

# Init space domain
dx = 1
x = np.arange(0, 100, dx)

# Init time domain
dt = 10                          # try 5 and 10 seconds (unstable)
start_time = np.datetime64("2021-01-01T00:00:00")
stop_time = np.datetime64("2021-01-01T20:00:00")
time = np.arange(start_time, stop_time, np.timedelta64(dt, 's'))

# Define initial conditions
u0 = np.cos(x * np.pi / x.max())

# This will be a diffusion problem, define the diffusion constant
k = 0.01 * np.ones_like(x)

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
ax = pde.plot_solution(x, time, solution)

# The exact solution to this situation
t_f = (stop_time - start_time).item().total_seconds() / 1  # Divide by 2 or 10 for comparison
exact = np.exp(-k[0] * t_f * np.pi**2 / x.max() ** 2) * np.cos(x * np.pi / x.max())
ax[0].plot(exact, x, 'g--', lw=2)

pt.bshow('test.png', save=False)
