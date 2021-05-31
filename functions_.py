import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as md
import matplotlib.cm as cm
from scipy.linalg.blas import dgbmv


def determine_dx_dt(x, time):
    """
    Get spatial and time resolutions from coordinates.

    Parameters
    ----------
    x : float or 1D array
        Returned as is if float, otherwise unique(diff(x)).
    time : float, 1D array of float or 1D array of numpy.datetime64
        Returned as is if float, otherwise unique(diff(x)). If
        `time` is a datetime array, the interval is returned in
        seconds.

    Returns
    -------
    dx, dt
        Model space and time steps.

    """
    # Determine x interval
    diff_x = np.diff(x).round(10)

    # x is not regularly spaced
    if np.unique(diff_x).size > 1:
        raise ValueError('x coordinate must be regularly spaced.')

    # x is regularly spaced
    else:
        dx = diff_x[0]

    # Time interval is directly given
    if isinstance(time, (int, float)):
        dt = time

    # Time is the coordinate
    else:
        diff_t = np.diff(time)

        # Time is not regularly spaced
        if np.unique(diff_t).size > 1:
            raise ValueError('time coordinate must be regularly spaced.')

        # Time is regularly spaced
        else:
            # Time is numeric
            if isinstance(diff_t[0], (int, float)):
                dt = diff_t[0]

            # Time is datetime64
            elif isinstance(diff_t[0], np.timedelta64):
                # Ensure it is in nanoseconds
                dt = diff_t[0].astype('timedelta64[ns]')

                # Convert to seconds
                dt = dt.item() / 10 ** 9

            # unsupported time format
            else:
                raise TypeError('time coordinate must be int, float or numpy datetime64.')

    return dx, dt


def banded_x_vector(matrix, x, ku=1, kl=1):
    """
    Matrix multiplication of banded matrix and vector.

    Parameters
    ----------
    matrix: 2D array
        Diagonal matrix stored in banded form.
    x: 1D array
        Vector with which to multiply.
    ku, kl: int
        Number of upper and lower diagonals in `matrix`

    Returns
    -------
    1D array
        Result of `full(matrix) * x`.

    """
    _, n = matrix.shape
    return dgbmv(n, n, kl, ku, 1, matrix, x)


def f_or_b(j_, diag, banded, ku=1):
    """
    Dim 0 index in either full or banded notations.

    Parameters
    ----------
    j_ : int
        Spatial coordinate index.
    diag: int
        Upper (1), center (0) or lower (-1) diagonal.

    Returns
    -------
    i_: int
        Dim 0 index for this element.
    """
    if diag not in [-1, 0, 1]:
            raise ValueError("diag must be -1, 0, or 1")
    if banded:
        # Middle diagonal
        if diag == 0:
            i_ = ku
        # upper diagonal
        elif diag == 1:
            i_ = ku + (j_ - 1) - j_
        # Lower diagonal
        else:
            i_ = ku + (j_ + 1) - j_
    else:
        if diag == 0:
            i_ = j_
        elif diag == 1:
            i_ = j_ - 1
        else:
            i_ = j_ + 1
    return i_


def matrix_full2banded(matrix):
    """
    Switch matrix from full to banded form.

    Parameters
    ----------
    matrix: 2D array
        Matrix stored in full form.

    Returns
    -------
    2D array
        Matrix stored in banded form.

    """
    m, n = matrix.shape
    banded = np.zeros((3, n))
    banded[0, 1:] = np.diagonal(matrix, 1)
    banded[1, :] = np.diagonal(matrix, 0)
    banded[2, :-1] = np.diagonal(matrix, -1)
    return banded


def coefs_implicit2explicit(coefs, alpha=-1):
    """
    Switch between coefs defined for implicit and explicit systems.

    Subtracts identity, multiplies by `alpha`, and adds identity. To
    switch between implicit and explicit `alpha` should be -1.

    Parameters
    ----------
    coefs: 2D array
        Matrix of coefficients to modify.
    alpha: float
        Amount by which to multiply `coefs - identity`

    Returns
    -------
    2D array
        Modified coefficients.

    """
    # Full matrix form
    if coefs.shape[0] == coefs.shape[1]:
        identity = np.eye(coefs.shape[0])

    # Banded matrix form
    else:
        identity = np.zeros((3, coefs.shape[-1]))
        identity[1, :] = 1.

    matrix = coefs.copy()
    matrix -= identity
    matrix *= alpha
    matrix += identity

    return matrix


def plot_solution(x, time, solution):
    """
    Visualize the output solution.

    Parameters
    ----------
    x, time : 1D array
        The space and time axes of the simulation.
    solution : 2D array
        Output of one of the `time_step` functions.

    Returns
    -------
    ax: 2-array of matplotlib.Axes
        The left and right panel axes.

    """
    # Initialize axes
    gskw = {'width_ratios': [1, 3], 'wspace': 0.2, 'bottom': 0.15, 'right': 1}
    _, ax = plt.subplots(1, 2, figsize=(8, 4), sharey=True, gridspec_kw=gskw)

    # Draw a the initial, 10%, 50% and final time steps 
    times_ = [0, time.size//10, time.size//2, time.size - 1]
    colors = ['k', 'r', 'r', 'b']
    styles = ['-', '--', '-', '-']
    labels = ['initial', r'10%', r'50%', 'final']
    for i_, c_, s_, l_ in zip(times_, colors, styles, labels):
        ax[0].plot(solution[:, i_], x, '%s%s' % (c_, s_), label=l_)
        ax[0].legend(fontsize='x-small')
        ax[0].set(ylabel='x')
        ax[0].tick_params(which='both', top=False, right=False)

    # Draw solution
    ctr = ax[1].pcolormesh(time,
                           x,
                           solution,
                           shading='auto',
                           cmap=cm.get_cmap('jet', 20))
    plt.colorbar(ctr)

    # Manage date axis
    length = (time[-1] - time[0]).item().total_seconds()
    if length < 60:
        fmt, xlabel = "%S", 'Seconds'
    elif length < 3600:
        fmt, xlabel = "%M:%S", 'Time (MM:SS)'
    elif length < 3600 * 24:
        fmt, xlabel = "%H:%M:%S", 'Time (HH:MM)'
    elif length < 3600 * 24 * 31:
        fmt, xlabel = "%d", 'Day'
    else:
        fmt, xlabel = "%y-%m%d", 'Date'
    fmt_obj = md.DateFormatter(fmt)
    ax[1].xaxis.set_major_formatter(fmt_obj)

    # Plot parameters
    ax[1].set(xlabel=xlabel, xlim=(time.min(), time.max()),
              ylim=(x.min(), x.max()))
    ax[1].tick_params(which='both', top=False, right=False)
    ax[1].tick_params(which='minor', bottom=False)

    return ax
