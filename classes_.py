import numpy as np
from scipy.linalg import solve, solve_banded
from .functions_ import *


class ImplicitSystem:
    """
    Physical parameters of the modeled system to be solved implicitly.

    This class stores the coefficients of the discretized system
    of equations as well as the boundary condition and forcing
    information. The system's domain is described below as 
    having `j` and `n` space and time steps.

    """

    def __init__(self, coefficients, boundary_0, boundary_j):
        """
        Parameters
        ----------
        coefficients: 2D array
            Of the discretized system of equations. Can be either
            an (j, j) tridiagonal matrix (K form) or a (3, j) matrix
            which is the banded form of K, as needed by
            `scipy.solve_banded`.
        boundary_0, boundary_j: float, 1D array or 3-tuple
            Values imposed on points x=0 and x=j. If float, the same
            value is applied at all time steps. If array, must contain
            `n` values i.e., one for each time step. If tuple, the
            boundaries are specified as a flux condition. The 2-tuple
            must contain the prescribed flux `Q` and spatial resolution
            `dx`.

        Note
        ----
        The flux must be Q in

        ..math::

            T_0 = T_1 + dx Q

        or similarly for the last grid point.

        ..math::

            T_j = T_{j-1} + dx Q

        """
        self.coefs = coefficients.copy()
        if isinstance(boundary_0, tuple):
            self.q0, self.dx = boundary_0
            self.boundary_0 = boundary_0
            self.qj, self.dx = boundary_j
            self.boundary_j = boundary_j
        else:
            self.boundary_0 = boundary_0
            self.boundary_j = boundary_j
        if self.coefs.shape[0] == self.coefs.shape[1]:
            self.banded_coefficients = False
            self.coefs_a_2 = self.coefs[1, 0]
            self.coefs_c_m1 = self.coefs[-2, -1]
        else:
            self.banded_coefficients = True
            self.coefs_a_2 = self.coefs[2, 0]
            self.coefs_c_m1 = self.coefs[0, -1]

    def coefs_bounded(self):
        """
        Return coefs with boundary adjustments.
        """
        return self.set_lhs_bounds(self.coefs)

    def set_lhs_bounds(self, matrix):
        """
        Adjust boundaries of LHS matrix before implicit solving.

        Parameters
        ----------
        matrix: 2D array
            Left hand side of the Ax = B system.

        Returns
        -------
        2D array
            Left hand side ready for implicit solving.

        """
        matrix = matrix.copy()
        if self.banded_coefficients:
            matrix[1, 0] = 1
            matrix[2, 0] = 0
            matrix[0, 1] = 0
            matrix[-2, -1] = 1
            matrix[-1, -2] = 0
            matrix[0, -1] = 0
        else:
            matrix[0, 0] = 1
            matrix[1, 0] = 0
            matrix[0, 1] = 0
            matrix[-1, -1] = 1
            matrix[-1, -2] = 0
            matrix[-2, -1] = 0
        return matrix

    def set_rhs_bounds(self, rhs, index=None):
        """
        Adjust right hand side (RHS) array for forcing and boundary conditions.

        Parameters
        ----------
        rhs: 1D array
            raw right hand side vector.
        index: int
            Number index of the current time step.

        Returns
        -------
        1D array
            Adjusted right hand side vector.

        """
        rhs = rhs.copy()
        if isinstance(self.boundary_0, (int, float)):
            rhs[0] = self.boundary_0
            rhs[-1] = self.boundary_j
        elif isinstance(self.boundary_0, np.ndarray):
            rhs[0] = self.boundary_0[index]
            rhs[-1] = self.boundary_j[index]
        elif isinstance(self.boundary_0, tuple):
            rhs[0] = rhs[1] + self.dx * self.q0
            rhs[-1] = rhs[-2] + self.dx * self.qj
        rhs[1] -= self.coefs_a_2 * rhs[0]
        rhs[-2] -= self.coefs_c_m1 * rhs[-1]

        return rhs

    def step_forward(self, rhs):
        """
        Move system one step forward in time.

        Parameters
        ----------
        rhs: 1D array
            Right hand side of Ax = B system before boundary adjustments.

        Returns:
        1D array:
            The solution of system at the next time step.

        """
        rhs = self.set_rhs_bounds(rhs)
        if self.banded_coefficients:
            next_step = solve_banded((1, 1), self.coefs_bounded(), rhs)
        else:
            next_step = solve(self.coefs_bounded(), rhs)

        return next_step


class ExplicitSystem:
    """
    Physical parameters of the modeled system to be solved explicitly.

    This class stores the coefficients of the discretized system
    of equations as well as the boundary condition and forcing
    information. The system's domain is described below as 
    having `j` and `n` space and time steps.

    """

    def __init__(self, coefficients, boundary_0, boundary_j):
        """
        Parameters
        ----------
        coefficients: 2D array
            Of the discretized system of equations. Can be either
            an (j, j) tridiagonal matrix (K form) or a (3, j) matrix
            which is the banded form of K, as needed by
            `scipy.solve_banded`.
        boundary_0, boundary_j: float, 1D array or 3-tuple
            Values imposed on points x=0 and x=j. If float, the same
            value is applied at all time steps. If array, must contain
            `n` values i.e., one for each time step. If tuple, the
            boundaries are specified as a flux condition. The 2-tuple
            must contain the prescribed flux `Q` and spatial resolution
            `dx`.

        Note
        ----
        The flux must be Q in

        ..math::

            T_0 = T_1 + dx Q

        or similarly for the last grid point.

        ..math::

            T_j = T_{j-1} + dx Q

        """
        self.coefs = coefficients.copy()
        if isinstance(boundary_0, tuple):
            self.q0, self.dx = boundary_0
            self.boundary_0 = boundary_0
            self.qj, self.dx = boundary_j
            self.boundary_j = boundary_j
        else:
            self.boundary_0 = boundary_0
            self.boundary_j = boundary_j
        if self.coefs.shape[0] == self.coefs.shape[1]:
            self.banded_coefficients = False
        else:
            self.banded_coefficients = True

    def set_solution_bounds(self, next_step, index=None):
        """
        Impose boundary conditions after calculating next step.

        Parameters
        ----------
        next_step: 1D array
            Solution of the coefficient and condition equation system..
        index: int
            Number index of the current time step.

        Returns
        -------
        1D array
            The next step with boundary conditions imposed.

        """
        next_step = next_step.copy()
        if isinstance(self.boundary_0, (int, float)):
            next_step[0] = self.boundary_0
            next_step[-1] = self.boundary_j
        elif isinstance(self.boundary_0, np.ndarray):
            next_step[0] = self.boundary_0[index]
            next_step[-1] = self.boundary_j[index]
        elif isinstance(self.boundary_0, tuple):
            next_step[0] = next_step[1] + self.dx * self.q0
            next_step[-1] = next_step[-2] + self.dx * self.qj

        return next_step

    def step_forward(self, current_step):
        if self.banded_coefficients:
            next_step = banded_x_vector(self.coefs, current_step)
        else:
            next_step = self.coefs @ current_step
        next_step = self.set_solution_bounds(next_step)
        return next_step
