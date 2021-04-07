"""Conjugate Gradient Method for solution of linear systems Ax = b with symmetric and positive definite matrix A."""
import numpy as np
import numpy.linalg as la

def cg(x0, A, b, tol = 1e-8, max_steps = 1000):
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
    rTr = np.inner(r,r)
    while la.norm(r) > tol and k < max_steps:
        Ap = A@p
        alpha = rTr/np.inner(p, Ap)
        x = x + alpha*p
        r1 = r + alpha*Ap
        beta = np.inner(r1, r1)/rTr
        p = -r1 + beta*p
        r = r1
        rTr = np.inner(r,r)
        k += 1
    return x, k, r
