""" Python code submission file.

IMPORTANT:
- Do not include any additional python packages.
- Do not change the existing interface and return values of the task functions.
- Prior to your submission, check that the pdf showing your plots is generated.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy.optimize as opt
from typing import Callable


# Modify the function bodies below to be used for function value and gradient computation
def func_1a(x: np.ndarray) -> float:
    """Computes and returns the function value for function 1d) at a given point x
    @param x Vector of size (2,)
    """

    """ Start of your code
    """
    x1 , x2 = x[0], x[1]
    f = 2*x1*x1*x1 - 6*x2*x2 + 3*x1*x1*x2    
    """ End of your code
    """
    return f


def grad_1a(x: np.ndarray) -> np.ndarray:
    """Computes and returns the analytical gradient result for function 1d) at a given point x
    @param x Vector of size (2,)
    """

    """ Start of your code
    """
    x1, x2 = x[0], x[1]
    partial_x1 = 6*x1*x1 + 6*x1*x2
    partial_x2 = -12*x2 + 3*x1*x1
    """ End of your code
    """
    return np.ndarray(shape=(2,), buffer=[partial_x1, partial_x2])


def func_1b(x: np.ndarray) -> float:
    """Computes and returns the function value for function 1d) at a given point x
    @param x Vector of size (2,)
    """

    """ Start of your code
    """
    x1, x2 = x[0], x[1]
    f =  x1*x1 + (x1 + 1)*(x1*x1 + x2*x2)
    """ End of your code
    """
    return f


def grad_1b(x: np.ndarray) -> np.ndarray:
    """Computes and returns the analytical gradient result for function 1d) at a given point x
    @param x Vector of size (2,)
    """

    """ Start of your code
    """
    x1, x2 = x[0], x[1]
    partial_x1 = 4*x1 + x2*x2 + 3*x1*x1
    partial_x2 = 2*x2*(x1 + 2)
    """ End of your code
    """
    return np.ndarray(shape=(2,), buffer=[partial_x1, partial_x2])


def func_1c(x: np.ndarray) -> float:
    """Computes and returns the function value for function 1d) at a given point x
    @param x Vector of size (2,)
    """

    """ Start of your code
    """
    x1, x2 = x[0], x[1]
    f = np.log(1 + 0.5 * (x1*x1 + 3*x2*x2*x2))
    """ End of your code
    """
    return f


def grad_1c(x: np.ndarray) -> np.ndarray:
    """Computes and returns the analytical gradient result for function 1d) at a given point x
    @param x Vector of size (2,)
    """

    """ Start of your code
    """
    x1, x2 = x[0], x[1]
    partial_x1 = (2*x1)/(2 + x1*x1 + 3*x2*x2*x2) 
    partial_x2 = (9*x2*x2)/(2 + x1*x1+3*x2*x2*x2)
    """ End of your code
    """
    return np.ndarray(shape=(2,0), buffer=[partial_x1, partial_x2])


def func_1d(x: np.ndarray) -> float:
    """Computes and returns the function value for function 1d) at a given point x
    @param x Vector of size (2,)
    """

    """ Start of your code
    """
    x1, x2 = x[0], x[1]
    f = (x1-2)*(x1-2) + x1*x2*x2 - 2
    """ End of your code
    """
    return f


def grad_1d(x: np.ndarray) -> np.ndarray:
    """Computes and returns the analytical gradient result for function 1d) at a given point x
    @param x Vector of size (2,)
    """

    """ Start of your code
    """
    x1, x2 = x[0], x[1]
    partial_x1 = 2*(x1-2) + x2*x2
    partial_x2 = 2*x1*x2
    """ End of your code
    """
    return np.ndarray(shape=(2,), buffer=[partial_x1, partial_x2])


def task1():
    """Characterization of Functions

    Requirements for the plots:
        - ax[0] Contour plot for a)
        - ax[1] Contour plot for b)
        - ax[2] Contour plot for c)
    Choose the number of contour lines such that the stationary points and the function can be well characterized.
    """
    print("\nTask 1")

    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle("Task 1 - Contour plots of functions", fontsize=16)

    ax[0].set_title("a)")
    ax[0].set_xlabel("$x_1$")
    ax[0].set_ylabel("$x_2$")

    ax[1].set_title("b)")
    ax[1].set_xlabel("$x_1$")
    ax[1].set_ylabel("$x_2$")

    ax[2].set_title("c)")
    ax[2].set_xlabel("$x_1$")
    ax[2].set_ylabel("$x_2$")

    ax[3].set_title("d)")
    ax[3].set_xlabel("$x_1$")
    ax[3].set_ylabel("$x_2$")

    """ Start of your code
    """

    """ End of your code
    """
    return fig


def task2():
    """Numerical Gradient Verification

    Implement the numerical gradient approximation using central differences in function approx_grad_task1. This function takes the function to be evaluated at point x as argument and returns the gradient approximation at that point.

    Pass the functions from task1 and compare the analytical and numerical gradient results for a given point x with np.allclose.

    Output the result of the comparison to the console.
    """
    print("\nTask 2")

    def approx_grad_task1(
        func: Callable, x: np.ndarray, eps: float, *args
    ) -> np.ndarray:
        """Numerical Gradient Computation
        @param x Vector of size (2,)
        @param eps float for numerical finite difference computation
        This function shall compute the gradient approximation for a given point 'x', 'eps' and a function 'func'
        using the given central differences formulation for 2D functions. (Task1 functions)
        @return The gradient approximation
        """

        """ Start of your code
        """

        """ End of your code
        """

        return np.zeros(2)

    """ Start of your code
    """

    """ End of your code
    """


# Modify the function bodies below to be used for function value and gradient computation
def func_3a(x: np.ndarray, A: np.ndarray, B: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Computes and returns the function value for function 3a) at a given point x
    @param x Vector of size (n,)
    @param A Matrix of size (m,o)
    @param B Matrix of size (o,n)
    @param b Vector of size (m,)
    """

    """ Start of your code
    """
    AB = A @ B
    v = np.dot(AB, x)
    v = v - b
    norm = np.linalg.norm(v)
    f = 0.5*norm*norm
    """ End of your code
    """
    return np.ndarray([f])


def grad_3a(x: np.ndarray, A: np.ndarray, B: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Computes and returns the gradient value for function 3a) at a given point x
    @param x Vector of size (n,)
    @param A Matrix of size (m,o)
    @param B Matrix of size (o,n)
    @param b Vector of size (m,)
    """

    """ Start of your code
    """

    """ End of your code
    """
    return np.zeros_like(x)


def hessian_3a(x):
    """Computes and returns the Hessian value for function 3a) at a given point x
    @param x Vector of size (n,)
    @param A Matrix of size (m,o)
    @param B Matrix of size (o,n)
    @param b Vector of size (m,)
    """

    """ Start of your code
    """

    """ End of your code
    """
    return np.zeros((x.shape[0], x.shape[0]))


def func_3b(x: np.ndarray, K: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Computes and returns the function value for function 3b) at a given point x
    @param x Vector of size (n,)
    @param K Matrix of size (n, n)
    @param t Vector of size (n,)
    """

    """ Start of your code
    """

    """ End of your code
    """
    return np.zeros(1)


def grad_3b(x: np.ndarray, K: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Computes and returns the gradient value for function 3b) at a given point x
    @param x Vector of size (n,)
    @param K Matrix of size (n, n)
    @param t Vector of size (n,)
    """

    """ Start of your code
    """

    """ End of your code
    """
    return np.zeros_like(x)


def hessian_3b(x: np.ndarray, K: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Computes and returns the Hessian value for function 3b) at a given point x
    @param x Vector of size (n,)
    @param K Matrix of size (n, n)
    @param t Vector of size (n,)
    """

    """ Start of your code
    """

    """ End of your code
    """
    return np.zeros((x.shape[0], x.shape[0]))


def func_3c(
    alpha: np.ndarray, A: np.ndarray, x: np.ndarray, y: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """Computes and returns the function value for function 3c) at a given point x
    @param a scalar of size (1,)
    @param A Matrix of size (m,n)
    @param x Vector of size (n,)
    @param y Matrix of size (n,)
    @param b Vector of size (m,)
    """

    """ Start of your code
    """

    """ End of your code
    """
    return np.zeros(1)


def grad_3c(
    alpha: np.ndarray, A: np.ndarray, x: np.ndarray, y: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """Computes and returns the gradient value for function 3c) at a given point x
    @param a scalar of size (1,)
    @param A Matrix of size (m,n)
    @param x Vector of size (n,)
    @param y Matrix of size (n,)
    @param b Vector of size (m,)
    """

    """ Start of your code
    """

    """ End of your code
    """
    return np.zeros_like(x)


def hessian_3c(
    alpha: np.ndarray, A: np.ndarray, x: np.ndarray, y: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """Computes and returns the Hessian value for function 3c) at a given point x
    @param a scalar of size (1,)
    @param A Matrix of size (m,n)
    @param x Vector of size (n,)
    @param y Matrix of size (n,)
    @param b Vector of size (m,)
    """

    """ Start of your code
    """

    """ End of your code
    """
    return np.zeros((x.shape[0], x.shape[0]))


def task3():
    """Matrix Calculus: Numerical Gradient Verification

    Utilize the function scipy.optimize.approx_fprime to numerically check the correctness of your analytic results. To this end, implement the functions func_3a, grad_3a, hessian_3a, func_3b, grad_3b, hessian_3b, func_3c, grad_3c, hessian_3c and compare them to the approximations.

    Check the correctness of your results by comparing the analytical and numerical results for three random points x with np.allclose. Also stick to the provided values for the present variables.

    Output the result of the comparison to the console.
    """
    print("\nTask 3")

    A = np.array([[0, 1], [2, 3]])  # do not change
    B = np.array([[3, 2], [1, 0]])  # do not change
    K = np.array([[1, 2], [2, 1]])  # do not change
    b = np.array([[4], [0.5]])  # do not change
    y = np.array([[1], [1]])  # do not change
    x = np.array([[0.5], [0.75]])  # do not change
    t = np.array([[7.5], [-3]])  # do not change

    """ Start of your code
    """

    """ End of your code
    """


def task4():
    """Linear Program

    Implement LP for student task selection and solve using scipy.optimize.linprog's solver.

    """
    print("\nTask 4")

    """ Start of your code
    """

    """ End of your code
    """


if __name__ == "__main__":
    pdf = PdfPages("figures.pdf")

    tasks = [task1, task2, task3, task4]

    for t in tasks:
        fig = t()

        if fig is not None:
            pdf.savefig(fig)

    pdf.close()
