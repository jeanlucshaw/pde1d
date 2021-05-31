import numpy as np
from .functions_ import *


def coefs_diffusion(x, time, k, implicit=True, banded=True):
    """
    Return coefficients for a diffusion system of equations.

    Parameters
    ----------
    x, time: float or 1D array
        Space and time coordinates.
    k: float or 1D array
        Diffusion constant over the spatial domain.
    implicit: bool
        Prepare coefficients for implicit solving.
    banded: bool
        Return coefficients in banded or full matrix form.

    Returns
    -------
    2D array
        Coefficients of the matrix representing the discretized
        diffusion equation.

    """
    # Coefficients are for an implicit or explicit system
    if implicit:
        switch = 1
    else:
        switch = -1

    # Domain steps
    dx, dt = determine_dx_dt(x, time)

    # Initialize
    if banded:
        K = np.zeros((3, x.size))
    else:
        # Block if K woudl be a big matrix
        if x.size > 1000:
            raise ValueError(
                'K matrix too large. Use implicit_banded_diffusion.')
        K = np.zeros((x.size, x.size))

    # Loop over time steps
    for j_ in np.arange(x.size):

        # Middle diagonal (b)
        K[f_or_b(j_, 0, banded), j_] = 1 + switch * 2 * dt * k[j_] / dx / dx

        # Lower diagonal
        if j_ < x.size - 1:
            if j_ == 0:
                k_term = k[j_] + 0.50 * (k[j_ + 1] - k[j_])
            else:
                k_term = k[j_] + 0.25 * (k[j_ + 1] - k[j_ - 1])
            K[f_or_b(j_, -1, banded), j_] = -1 * switch * dt * k_term / dx / dx

        # Upper diagonal
        if j_ > 0:
            if j_ == x.size - 1:
                k_term = k[j_] - 0.50 * (k[j_] - k[j_ - 1])
            else:
                k_term = k[j_] - 0.25 * (k[j_ + 1] - k[j_ - 1])
            K[f_or_b(j_, 1, banded), j_] = -1 * switch * dt * k_term / dx / dx

    return K
