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
    # return np.ndarray(shape=(2,), buffer=[partial_x1, partial_x2])
    return np.array([partial_x1, partial_x2])


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
    partial_x2 = 2*x2*(x1 + 1)
    """ End of your code
    """
    return np.array([partial_x1, partial_x2])


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
    return np.array([partial_x1, partial_x2])


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
    return np.array([partial_x1, partial_x2])


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
    x1, x2 = np.meshgrid(np.linspace(-6, 2, 100), np.linspace(-2,6,100))
    cs = ax[0].contour(x1,x2,func_1a([x1,x2]),levels=125)
    ax[0].clabel(cs, inline=1, fontsize=10)
    ax[0].plot(0,0,'*',label='saddle point')
    ax[0].plot(-4,4,'*',label='saddle point')

    x1, x2 = np.meshgrid(np.linspace(-2, 1, 200), np.linspace(-2,2,100))
    cs = ax[1].contour(x1,x2,func_1b([x1,x2]),levels=200)
    ax[1].clabel(cs, inline=1, fontsize=10)
    ax[1].plot(-4.0/3.0, 0, '*',label='saddle point')
    ax[1].plot(-1,1,'*',label='saddle point')
    ax[1].plot(-1,-1,'*',label='saddle point')
    ax[1].plot(0,0,'*',label='local minimum')

    x1, x2 = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-0.5,4,100))
    cs = ax[2].contour(x1,x2,func_1c([x1,x2]),levels=100)
    ax[2].clabel(cs, inline=1, fontsize=10)
    ax[2].plot(0,0,'*',label='saddle point')

    x1, x2 = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-4,4,100))
    cs = ax[3].contour(x1,x2,func_1d([x1,x2]),levels=125)
    ax[3].clabel(cs, inline=1, fontsize=10)
    ax[3].plot(2,0,'*',label='local minimum')
    ax[3].plot(0,2,'*',label='saddle point')
    ax[3].plot(0,-2,'*',label='saddle point')

    for a in ax:
        a.legend()
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
        x1, x2 = x

        approx_grad_x1 = (func([x1 + eps, x2]) - func([x1 - eps, x2])) / (2 * eps)
        approx_grad_x2 = (func([x1, x2 + eps]) - func([x1, x2 - eps])) / (2 * eps)

        """ End of your code
        """

        return np.array([approx_grad_x1, approx_grad_x2])

    """ Start of your code
    """
    eps =  1e-03
    num_steps = 100000
    random_points = np.linspace(0, 10, num_steps) #meant to simulate the set R has to be positive because of log
    random_indices = np.random.choice(len(random_points), 6, replace=False) #
    
    x1_samples = [random_points[i] for i in random_indices[:3]]
    x2_samples = [random_points[i] for i in random_indices[3:]]

    points = [[x1, x2] for x1, x2 in zip(x1_samples, x2_samples)]
    
    assert len(x1_samples) == len(x2_samples)
    print('Points: {}'.format(points))
    print('========================')
    
    approximations_list = []

    for point in points:
        print('Point: {}\n'.format(point))
        approx_g_a = approx_grad_task1(func_1a, point, eps)
        approx_g_b = approx_grad_task1(func_1b, point, eps)
        approx_g_c = approx_grad_task1(func_1c, point, eps)
        approx_g_d = approx_grad_task1(func_1d, point, eps)

        g_a = grad_1a(point)
        g_b = grad_1b(point)
        g_c = grad_1c(point)
        g_d = grad_1d(point)

        print('Gradient - Approximation')
        print('a) {} - {}'.format(g_a, approx_g_a))
        print('b) {} - {}'.format(g_b, approx_g_b))
        print('c) {} - {}'.format(g_c, approx_g_c))
        print('d) {} - {}'.format(g_d, approx_g_d))

        approximations_list.append([g_a, approx_g_a, point])
        approximations_list.append([g_b, approx_g_b, point])
        approximations_list.append([g_c, approx_g_c, point])
        approximations_list.append([g_d, approx_g_d, point])

        print('========================')
    
    for gradient, approximation, point in approximations_list:
        assert np.allclose(gradient, approximation, rtol = 0, atol = 10*eps**2), f"Gradient {gradient} and approximation {approximation} do not match for point {point}"
    
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
    v = AB @ x
    v = v - b
    norm = np.linalg.norm(v)
    f = 0.5*np.square(norm)
    out = np.array([f])
    
    """ End of your code
    """

    return out


def grad_3a(x: np.ndarray, A: np.ndarray, B: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Computes and returns the gradient value for function 3a) at a given point x
    @param x Vector of size (n,)
    @param A Matrix of size (m,o)
    @param B Matrix of size (o,n)
    @param b Vector of size (m,)
    """

    """ Start of your code
    """

    AB = A @ B
    ABT = AB.T
    v = AB @ x
    v = v - b
    grad = ABT @ v
    
    """ End of your code
    """
    return grad


def hessian_3a(x: np.ndarray, A: np.ndarray, B: np.ndarray, b: np.ndarray):
    """Computes and returns the Hessian value for function 3a) at a given point x
    @param x Vector of size (n,)
    @param A Matrix of size (m,o)
    @param B Matrix of size (o,n)
    @param b Vector of size (m,)
    """

    """ Start of your code
    """
    AB = A@B
    ABT = AB.T
    hess = ABT @ AB
    """ End of your code
    """
    return hess


def func_3b(x: np.ndarray, K: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Computes and returns the function value for function 3b) at a given point x
    @param x Vector of size (n,)
    @param K Matrix of size (n, n)
    @param t Vector of size (n,)
    """

    """ Start of your code
    """
    ones = np.ones(x.size)
    x1 = ones @ x
    xt = x*t
    xtt = xt.T
    f = x1 - 0.5*(xtt @ (K @ xt))
    """ End of your code
    """
    return f


def grad_3b(x: np.ndarray, K: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Computes and returns the gradient value for function 3b) at a given point x
    @param x Vector of size (n,)
    @param K Matrix of size (n, n)
    @param t Vector of size (n,)
    """

    """ Start of your code
    """

    ones = np.ones(x.size)
    xt = x*t
    Kxt = K @ xt
    tKxt = t*Kxt
    grad = ones - tKxt
    """ End of your code
    """
    return grad


def hessian_3b(x: np.ndarray, K: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Computes and returns the Hessian value for function 3b) at a given point x
    @param x Vector of size (n,)
    @param K Matrix of size (n, n)
    @param t Vector of size (n,)
    """

    """ Start of your code
    """
    T = np.outer(t, t)
    KT = K*T
    hessian = -KT
    """ End of your code
    """
    return hessian


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

    ay = alpha*y
    xay = x - ay
    Axay = A @ xay
    Axay = Axay - b
    f = 0.5*np.square(np.linalg.norm(Axay))
    """ End of your code
    """
    return f


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
    ay = alpha*y
    
    xay = x - ay
    Axay = A @ xay
    Axay = Axay - b
    Axay = Axay.T
    Ay = A@y
    grad = - Axay @ Ay
    """ End of your code
    """
    return grad


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
    Ay = A @ y
    Ay_transpose = Ay.T
    hess = Ay_transpose @ Ay
    """ End of your code
    """
    return hess


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

    bflatten = np.ndarray.flatten(b)
    tflatten = np.ndarray.flatten(t)
    xflatten = np.ndarray.flatten(x)
    yflatten = np.ndarray.flatten(y)

    def partial(x, f:Callable, i:int, *args):
        return f(x, *args)[i]
    
    approximation_list = []
    eps = 1E-6
    
    for i in range(3):
        x1_random = np.random.rand(2,) * 10
        x2_random = np.random.rand(2,) * 10
        alpha_random = np.random.rand(1,) * 10


        approx_grad_a = opt.approx_fprime(x1_random, func_3a, eps, A, B, bflatten)
        approx_grad_b = opt.approx_fprime(x2_random, func_3b, eps, K, tflatten)
        approx_grad_c = opt.approx_fprime(alpha_random, func_3c, eps, A, xflatten, yflatten, bflatten)
        grad_3a_true = grad_3a(x1_random,A,B,bflatten)
        grad_3b_true = grad_3b(x2_random,K,tflatten)
        grad_3c_true = grad_3c(alpha_random, A, xflatten, yflatten, bflatten)

        approximation_list.append([grad_3a_true, approx_grad_a, x1_random])
        approximation_list.append([grad_3b_true, approx_grad_b, x2_random])
        approximation_list.append([grad_3c_true, approx_grad_c, alpha_random])

        approx_hess_a = np.concatenate((opt.approx_fprime(x1_random, partial, eps, grad_3a, 0 ,A,B, bflatten),opt.approx_fprime(x1_random, partial, eps, grad_3a, 1 ,A,B, bflatten)), axis=None)
        hess_true_a = np.ndarray.flatten(hessian_3a(x1_random, A, B, bflatten))
        approx_hess_b = np.concatenate((opt.approx_fprime(x2_random, partial, eps, grad_3b, 0, K,tflatten), opt.approx_fprime(x2_random, partial, eps, grad_3b, 1, K,tflatten)), axis=None)
        hess_true_b = np.ndarray.flatten(hessian_3b(x2_random,K,tflatten))
        approx_hess_c = opt.approx_fprime(alpha_random, grad_3c, eps, A,xflatten,yflatten,bflatten)
        hess_true_c = hessian_3c(alpha_random, A,xflatten,yflatten,bflatten)

        approximation_list.append([hess_true_a, approx_hess_a, x1_random])
        approximation_list.append([hess_true_b, approx_hess_b, x2_random])
        approximation_list.append([hess_true_c, approx_hess_c, alpha_random])

        print("\nApproximation {}:".format(i + 1))
        print("a) x =", x1_random, '\nTrue Gradient:', grad_3a_true, '\nApproximation:', approx_grad_a)
        print("a) x =", x1_random, '\nTrue Hessian:', hess_true_a, '\nApproximation:', approx_hess_a)
        print("b) x =", x2_random, '\nTrue Gradient:', grad_3b_true, '\nApproximation:', approx_grad_b)
        print("b) x =", x2_random, '\nTrue Hessian:', hess_true_b, '\nApproximation:', approx_hess_b)
        print("c) alpha = ", alpha_random, '\nTrue Gradient:', grad_3c_true , '\nApproximation:', approx_grad_c)
        print("c) alpha = ", alpha_random, '\nTrue Hessian:', hess_true_c, '\nApproximation:', approx_hess_c)


        for true_val, approximation, point in approximation_list:            
            assert np.allclose(true_val, approximation, rtol=0, atol=100*eps ), f"True value {true_val} and approximation {approximation} do not match for point {point}"


    """ End of your code
    """


def task4():
    """Linear Program

    Implement LP for student task selection and solve using scipy.optimize.linprog's solver.

    """
    print("\nTask 4")

    """ Start of your code
    """

    costs_1_s =  [0.5, 0.25, 0.25, 0.25, 1.0, 1.0, 0.5, 0.5, 1.0, 0.5, 1.5, 2.5, 1.0, 2.5, 3.5, 9.0]
    costs_2_s = [0.75, 1.0, 0.75, 0.5, 0.5, 0.25, 0.25, 0.25, 2.0, 1.25, 1.0, 4.0, 2.5, 3.0, 2.0, 6.0]
    c_s = [0.5, 0.25, 0.25, 0.25, 1.0, 1.0, 0.5, 0.5, 1.0, 0.5, 1.5, 2.5, 1.0, 2.5, 3.5, 0.0,
     0.75, 1.0, 0.75, 0.5, 0.5, 0.25, 0.25, 0.25, 2.0, 1.25, 1.0, 4.0, 2.5, 3.0, 2.0, 0.0]
    A_eq_s = [np.append(costs_1_s, np.zeros(16)), np.append(np.zeros(16), costs_2_s)]
    for i in range(15):
        arr = np.zeros(32)
        arr[i] = 1
        arr[i + 16] = 1
        A_eq_s.append(arr)
    b_eq_s = [9, 6] + [1] * 15


    A_eq_s = np.array(A_eq_s)

    bounds_s = [(0, 1) for _ in range(32)]
    bounds_s[15]=(0, 1e20)
    bounds_s[31]=(0, 1e20)

    sol_s = opt.linprog(c_s, A_eq=A_eq_s, b_eq=b_eq_s, bounds=bounds_s)['x']
    out = np.sum(sol_s@c_s)
    print('Minimum total time spent: {} hours'.format(out)) #final answer
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
