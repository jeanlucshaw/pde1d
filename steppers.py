import numpy as np

def time_step_euler(initial, time, system):
    """
    Step `system` through `time` from `initial` conditions.

    Parameters
    ----------
    initial: 1D array
        Initial conditions of the experiment.
    time: 1D array
        Time coordinate of the experiment.
    system: ImplicitSystem or ExplicitSystem
        Object decribing the system of equations and boundary conditions.

    Returns
    -------
    solution: 2D array
        Containing the time stepped equations.

    """
    # Init output
    solution = np.nan * np.zeros((initial.size, time.size))

    # Set initial conditions
    solution[:, 0] = initial

    # Time step
    for i_ in range(time.size)[1:]:
        solution[:, i_] = system.step_forward(solution[:, i_ - 1])

    return solution


def time_step_theta(initial, time, system, theta):
    """
    Step `system` through `time` from `initial` conditions.

    Apply to implicit systems. The `theta` parameter acts as
    a continuous control (0-1) switching between the Euleur
    explicit (0), Cranck-Nicholson (0.5), and Euler implicit
    finite difference solver schemes.

    Parameters
    ----------
    initial: 1D array
        Initial conditions of the experiment.
    time: 1D array
        Time coordinate of the experiment.
    system: ImplicitSystem
        Object decribing the system of equations and boundary conditions.

    Returns
    -------
    solution: 2D array
        Containing the time stepped equations.

    """

    # Init output
    solution = np.nan * np.zeros((initial.size, time.size))

    # Set initial conditions
    solution[:, 0] = initial

    # Special case: empty LHS => explicit solving
    if theta == 0:
        # Switch coefficient signs (n+1 --> n)
        coefs = coefs_implicit2explicit(system.coefs, -1)

        # Prepare system for explicit solution
        theta_system = ExplicitSystem(coefs, system.boundary_0, system.boundary_j)

    # General case: coefficients on the LHS and RHS => implicit solving
    else:
        """
        The implicit2explicit function effectively switches the coefficients from
        the LHS to the RHS and vice versa. This is why it can be used to move a
        fraction of the coefficients to the RHS in proportions determined by `theta`.
        """
        # Apply theta to hand side matrix
        rhs_coefs = coefs_implicit2explicit(system.coefs, -1 * (1 - theta)) 

        # Calculate left hand side
        lhs_coefs = coefs_implicit2explicit(system.coefs, theta)

        # Prepare 
        theta_system = ImplicitSystem(lhs_coefs, system.boundary_0, system.boundary_j)

    # Time step
    for i_ in range(time.size)[1:]:

        # Euler explicit < --
        if isinstance(theta_system, ExplicitSystem):
            rhs = solution[:, i_ - 1]

        # -- > Cranck-Nicholson < -- > Euler implicit
        else:
            # Calculate right hand side
            if theta_system.banded_coefficients:
                rhs = banded_x_vector(rhs_coefs, solution[:, i_ -1])
            else:
                rhs = rhs_coefs @ solution[:, i_ -1]

        # Step forward
        solution[:, i_] = theta_system.step_forward(rhs)

    return solution
