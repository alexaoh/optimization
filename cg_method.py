"""Conjugate Gradient Method for solution of linear systems Ax = b with symmetric and positive definite matrix A."""
import numpy as np
import numpy.linalg as la

def cg(x0, A, b, tol, max_steps = 1000):
    """Conjugate Gradient Method.
    
    Input:
    x0: Initial guess of solution.
    A: Matrix of left side.
    b: Vector of right side.
    tol: Tolerance of \ell^2-norm of residuals. 
    max_steps: Maximal amount of iterations allowed. 

    Output: 
    x: Solution.
    k: Number of iterations. 
    r: Residual in final step. 
    """
    r = A@x0-b
    p = -r
    k = 0
    x = x0
    while la.norm(r) > tol and k < max_steps:
        alpha = np.inner(r,r)/np.inner(p, A@p)
        x = x + alpha*p
        r1 = r + alpha*A@p
        beta = np.inner(r1, r1)/np.inner(r, r)
        p = -r1 + beta*p
        k += 1
        r = r1
    return x, k, r
