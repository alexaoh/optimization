"""Problem 1 and 2 on Exercise Set 4 in Numerical Optimization."""
import numpy as np
from cg_method import cg
from steepest_descent import steepest_descent

# Problem 1. 
# Implement both the gradient descent method and Newton’s method with a line search satisfying the Wolfe conditions 
# (you may want to use a bisection algorithm for the implementation of these conditions).
# Apply your method to the minimisation of the Rosenbrock function,

def f(x):
    """Rosenbrock function."""
    assert(x.shape == (2,))
    return 100*(x[1] - x[0]**2)**2 + (1-x[0])**2

def grad_f(x):
    """Gradient of the Rosenbrock function."""
    assert(x.shape == (2,))
    return np.array([-400*(x[1] - x[0]**2)*(x[0]) - (1-x[0]), 200*(x[1] - x[0]**2)])


# Need to complete the implementation of the steepest descent and begin implementing Newton's method. This should be very similar!


# Problem 2.
# Solve the system below with CGM.

A = np.array([
    [2, -1, -1], 
    [-1, 3, -1], 
    [-1, -1, 2]])

b = np.array([1, 0, 1])
x0 = np.zeros(A.shape[0])

x, iterations, res = cg(x0, A, b, 1e-10)
print(x, iterations, res)
# It is apparent that the method converges after two steps, with the exact solution x = [3, 2, 3].
