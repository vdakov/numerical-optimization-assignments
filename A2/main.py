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

    #contour_levels = 10
    #x1, x2 = np.meshgrid(np.linspace(-lim, lim), np.linspace(-lim, lim))
    #objective = x1 - 2 * x2  # some arbitrary objective function
    #equality_connstraint = -0.5 * x1 - x2 - 2  # example equality constriant (=0)
    #inequality_constraint = -x1 + x2 + 2  # example inequality constriant (<=0)
#
    ## plot the level lines of the objective function and add labels to the lines
    #contours_objective = ax[0].contour(x1, x2, objective, contour_levels)
    #ax[0].clabel(contours_objective, fmt="%2.1f", use_clabeltext=True)
#
    ## plot the equality constraint by using ax.contour with the level line at 0
    #constraint_color = "orangered"
    #ax[0].contour(x1, x2, equality_connstraint, [0], colors=constraint_color)
#
    ## plot the inequality constraint in the same way but also add indicator for the feasible region
    #feasible_region_indicator = patheffects.withTickedStroke(angle=-90, length=1)
    #contours_inequality = ax[0].contour(
    #    x1, x2, inequality_constraint, [0], colors=constraint_color
    #)
    #contours_inequality.set(path_effects=[feasible_region_indicator])
#
    ## plot some (arbitrary) candidate points
    #color_optimal = "green"
    #marker_optimal = "*"
    #ax[0].scatter(0, -2, c=color_optimal, marker=marker_optimal, zorder=2)
    #color_valid = "black"
    #marker_valid = "o"
    #ax[0].scatter(2, -3, c=color_valid, marker=marker_valid, zorder=2)
    #color_invalid = "red"
    #marker_invalid = "x"
    #ax[0].scatter(1, 1, c=color_invalid, marker=marker_invalid, zorder=2)

    # EXAMPLE END

    constraint_color = 'orange'
    color_optimal = 'green'
    color_invalid = 'red'
    color_valid = 'black'
    
    marker_optimal = '*'
    marker_invalid = 'x'
    marker_valid = 'o'

    x1, x2 = np.meshgrid(np.linspace(-lim, lim), np.linspace(-lim, lim))
    contour_levels = 10

    function_a = -x1 + 2*x2
    eq_constraint_a = -x1 - 0.5*x2*x2 + 3 
    ineq_constraint_a = -0.5*x2 -3*x1 + 5

    function_b = 3*x2 + 0.33*x1*x1
    eq_constraint_b = (1/x2)*(x1*x1 - 3) - 1
    ineq_constraint_b = 0.5*x1 - 1 - x2

    function_c = 0.5*(2*x1*x1 + x1*x2 + 4*x2*x2)
    eq_constraint_c = 0.5*x1 - x2 - 1
    ineq_constraint_c = x1*x1 + x2 - 3
    
    # Plot A
    contours_objective = ax[0].contour(x1, x2, function_a, contour_levels)
    ax[0].clabel(contours_objective, fmt="%2.1f", use_clabeltext=True)
    ax[0].contour(x1, x2, eq_constraint_a, [0], colors=constraint_color)
    feasible_region_indicator = patheffects.withTickedStroke(angle=-90, length=1)
    contours_inequality = ax[0].contour(x1, x2, ineq_constraint_a, [0], colors=constraint_color)
    contours_inequality.set(path_effects=[feasible_region_indicator])

    ax[0].scatter(1,-2, c=color_valid, marker=marker_invalid)
    ax[0].scatter(1.36, 1.808, c=color_valid, marker=marker_invalid)
    ax[0].scatter(1.91 , -1.474, c=color_valid, marker=marker_valid)

    ax[0].scatter(1.91 , -1.474, c=color_optimal, marker=marker_optimal)

    # Plot B
    contours_objective = ax[1].contour(x1, x2, function_b, contour_levels)
    ax[1].clabel(contours_objective, fmt="%2.1f", use_clabeltext=True)
    ax[1].contour(x1, x2, eq_constraint_b, [0], colors=constraint_color)
    feasible_region_indicator = patheffects.withTickedStroke(angle=-90, length=1)
    contours_inequality = ax[1].contour(x1, x2, ineq_constraint_b, [0], colors=constraint_color)
    contours_inequality.set(path_effects=[feasible_region_indicator])
    
    ax[1].scatter(0, -3, c=color_valid, marker=marker_invalid)
    ax[1].scatter(1.68, -0.15, c=color_valid, marker=marker_valid)
    ax[1].scatter(-1.18614, -1.59307, c=color_valid, marker=marker_valid)
    

    ax[1].scatter(-1.18614, -1.59307, c=color_optimal, marker=marker_optimal)


    # Plot C
    contours_objective = ax[2].contour(x1, x2, function_c, contour_levels)
    ax[2].clabel(contours_objective, fmt="%2.1f", use_clabeltext=True)
    ax[2].contour(x1, x2, eq_constraint_c, [0], colors=constraint_color)
    feasible_region_indicator = patheffects.withTickedStroke(angle=-90, length=1)
    contours_inequality = ax[2].contour(x1, x2, ineq_constraint_c, [0], colors=constraint_color)
    contours_inequality.set(path_effects=[feasible_region_indicator])


    ax[2].scatter(0.714, -0.642, c=color_valid, marker=marker_valid)
    ax[2].scatter(1.76, -0.11, c=color_invalid, marker=marker_invalid)
    ax[2].scatter(-2.26, -2.13, c=color_invalid, marker=marker_invalid)

    ax[2].scatter(0.714, -0.642, c=color_optimal, marker=marker_optimal)
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
    contour_levels = [0, 0.1, 0.25, 0.5, 0.7, 1, 2, 3, 4, 5.5]
    x1, x2 = np.meshgrid(np.linspace(0.9, 8), np.linspace(0, 5))
    
    objective = x2/x1
    h = a + b*(x2-c)*(x2-c) - x1

    ax.contour(x1, x2, h, [0], colors='orange')
    contour = ax.contourf(x1,x2,objective,levels=contour_levels)
    ax.scatter(2,  1.5, c='blue', marker='x', label='Graphical opt')  
    ax.scatter(2, np.sqrt(2), c='red', marker='*', label='Analytical opt')  
    plt.colorbar(contour)

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
