"""Gradient descent method with a line search satisfying the Wolfe conditions."""

def backtracking_line_search(f, grad_f, p, alpha_bar, rho, c, x):
    """Backtracking line search to satisfy the Wolfe conditions.

    Input: 
    f: Objective function.
    grad_f: Gradient of the objective function at the current iterate x. 
    p: Step direction.
    alpha_bar: Initial step length.
    rho: Contraction factor in each disregarded step length.
    c: Constant in first Wolfe condition.
    x: Current iterate.

    Output:
    alpha: Step length to satisfy the Wolfe conditions. 
    """
    # grad_f needs to be callable here as well, in order to match the functionality of the bisecting line search!
    alpha = alpha_bar
    while f(x+alpha*p) > f(x) + c*alpha*np.inner(grad_f, p):
        alpha *= rho
    return alpha

def bisecting_line_search(f, grad_f, p, rho, c_1, c_2, x):
    """Line search using bisection to satisfy the Wolfe conditions.

    Input: 
    f: Objective function.
    grad_f: Callable gradient of the objective function at the current iterate x. 
    p: Step direction.
    rho: Contraction factor in each disregarded step length.
    c_1: Constant in first Wolfe condition.
    c_2: Constant in second Wolfe condition.
    x: Current iterate.

    Output:
    alpha: Step length to satisfy the Wolfe conditions. 
    """
    if not callable(grad_f):
        print("The gradient of the objective function needs to be a callable function!")
        return

    alpha = 1
    alpha_min = 0
    alpha_max = float("inf")
    
    # Should perhaps add a maxiter condition? Check if necessary later!
    while True:
        if f(x+alpha*p) > f(x) + c_1*alpha*np.inner(grad_f(x), p): # No sufficient decrease.
            a_max = alpha
            alpha = (alpha_min + alpha_max)/2
        elif np.inner(grad_f(x + alpha*p), p) < c_2*np.inner(grad_f(x), p): # No curvature condition. 
            alpha_min = alpha
            if alpha_max == float("inf"):
                alpha *= 2
            else: 
                alpha = (alpha_min + alpha_max)/2
        else:
            return alpha


def steepest_descent(x0, f, grad_f, tol, line_search_method):
    """Steepest descent method with (backtracking or bisection) line search satisfying the Wolfe conditions.
    
    Input:
    x0: Starting guess. 
    f: Callable objective function. 
    grad_f: Callable function returning the gradient of the objective function.
    tol: Tolerance of the size of the gradient at the current iterate. 
    line_search_method: Callable line search method function.  

    Output:
    x: Approximate solution found using steepest descent. 
    k: Iterations used. 
    """
    if not callable(line_search_method):
        print("The last argument needs to be a callable line search function!")
        return 
    if not callable(grad_f):
        print("The gradient of the objective function needs to be a callable function!")
        return 
    if not callable(f):
        print("The objective function must be a callable function!")
        return 

    x = x0
    k = 0
    gradient = grad_f(x) # In order to save some function calls. 
    while k < maxiter and gradient > tol:
        alpha = line_search_method(f, grad_f, p, alpha_bar, rho, c, x)
        x -= alpha*gradient
        gradient = grad_f(x)
        k += 1 
    return x, k
