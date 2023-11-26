#!/usr/bin/env python

""" Python code submission file.

IMPORTANT:
- Do not include any additional python packages.
- Do not change the existing interface and return values of the task functions.
- Prior to your submission, check that the pdf showing your plots is generated.
"""

from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.core.fromnumeric import shape
from scipy.linalg import inv
from matplotlib.backends.backend_pdf import PdfPages
from typing import Callable
from matplotlib import patheffects

import numpy as np


def task1():
    """Lagrange Multiplier Problem

    Requirements for the plots:
        - ax[0] Contour plot for a)
        - ax[1] Contour plot for b)
        - ax[2] Contour plot for c)
    """

    lim = 5

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Task 1 - Contour plots + Constraints", fontsize=16)

    for title, a in zip(["a)", "b)", "c)"], ax):
        a.set_title(title)
        a.set_xlabel("$x_1$")
        a.set_ylabel("$x_2$")
        a.set_aspect("equal")
        a.set_xlim([-lim, lim])
        a.set_ylim([-lim, lim])

    """ Start of your code
    """
    # Plot example (remove in submission)
    # EXAMPLE START

    contour_levels = 10
    x1, x2 = np.meshgrid(np.linspace(-lim, lim), np.linspace(-lim, lim))
    objective = x1 - 2 * x2  # some arbitrary objective function
    equality_connstraint = -0.5 * x1 - x2 - 2  # example equality constriant (=0)
    inequality_constraint = -x1 + x2 + 2  # example inequality constriant (<=0)

    # plot the level lines of the objective function and add labels to the lines
    contours_objective = ax[0].contour(x1, x2, objective, contour_levels)
    ax[0].clabel(contours_objective, fmt="%2.1f", use_clabeltext=True)

    # plot the equality constraint by using ax.contour with the level line at 0
    constraint_color = "orangered"
    ax[0].contour(x1, x2, equality_connstraint, [0], colors=constraint_color)

    # plot the inequality constraint in the same way but also add indicator for the feasible region
    feasible_region_indicator = patheffects.withTickedStroke(angle=-90, length=1)
    contours_inequality = ax[0].contour(
        x1, x2, inequality_constraint, [0], colors=constraint_color
    )
    contours_inequality.set(path_effects=[feasible_region_indicator])

    # plot some (arbitrary) candidate points
    color_optimal = "green"
    marker_optimal = "*"
    ax[0].scatter(0, -2, c=color_optimal, marker=marker_optimal, zorder=2)
    color_valid = "black"
    marker_valid = "o"
    ax[0].scatter(2, -3, c=color_valid, marker=marker_valid, zorder=2)
    color_invalid = "red"
    marker_invalid = "x"
    ax[0].scatter(1, 1, c=color_invalid, marker=marker_invalid, zorder=2)

    # EXAMPLE END



    """ End of your code
    """

    return fig


def task2():
    """Glider Trajectory Problem

    Requirements for the plot (only main ax):
        - filled contour plot for objective function
        - contour plot for constraint at level set 0
        - mark graphically estimated optimum
        - mark analytically determined optimum
    """

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    fig.suptitle("Task 2 - Glider's Trajectory", fontsize=16)

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")

    a = 1.0
    b = 0.5
    c = 0.0

    """ Start of your code
    """



    """ End of your code
    """

    plt.legend()
    return fig


if __name__ == "__main__":
    tasks = [task1, task2]

    pdf = PdfPages("figures.pdf")
    for task in tasks:
        retval = task()
        fig = retval[0] if type(retval) is tuple else retval
        pdf.savefig(fig)
    pdf.close()
