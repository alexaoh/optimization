import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg as la
from scipy.spatial import ConvexHull,convex_hull_plot_2d
from matplotlib import rc
import time as t

def Test(k, V, A):
    """Check if a^k is a global minimizer."""
    M = np.arange(len(V)); M = M[M!=k]
    return np.sqrt(np.sum([np.sum([V[i] * (A[k][j]-A[i][j]) / (la.norm(A[k]-A[i])) for i in M])**2 for j in [0, 1]]))

def f_grad(x, V, A):
    """Gradient of objective function."""
    M = np.arange(len(V))
    return np.array([[np.sum([V[i]*(A[i][j]-x[j])/(la.norm(A[i]-x)) for i in M])] for j in [0, 1]])

def f_d2(x, V, A):
    """Objective function."""
    M = np.arange(len(V))
    return np.sum([V[i] * la.norm(A[i]-x) for i in M])

def starting_point(V, A):
    """Starting point found by solving the Median problem using squared Euclidean norm."""
    return np.average(A, 0)

def sigma(x, A):
    """Sigma as defined in 3.1 in problem description."""
    hull = ConvexHull(A)
    #points = np.array([A[hull.vertices,0], A[hull.vertices,1]]).T
    points = A[hull.vertices] # Dette gir ut det samme som ovenfor, velg en av dem ;)
    return np.max([la.norm(x-y) for y in points])

def stop(x, V, A, eps):
    """Termination criterion for the Weiszfeld algorithm based on Theorem 4."""
    LB = f_d2(x, V, A)-la.norm(f_grad(x, V, A)*sigma(x, A))
    if LB > 0:
    	return (la.norm(f_grad(x, V, A)) * sigma(x, A)) / LB<=eps
    raise ZeroDivisionError

def Weiszfeld(V, A, eps=1e-14, max_iter=10000):
    """Weiszfeld algorithm for solving the Median problem with weighted Euclidean distance."""
    assert len(V)==len(A)
    M = np.arange(len(V))
    for k in M:
        if Test(k, V, A)<=V[k]:
            print("Minimum attained already")
            return A[k]
    x = starting_point(V, A)
    print(x, "starting point")
    its = []
    obj_vals = []
    times = []
    t1 = t.time()
    for _ in range(max_iter):
        if stop(x, V, A, eps):
            break
        x = np.array([np.sum([V[i]*A[i][j]/la.norm(A[i]-x) for i in M]) / np.sum([V[i]/la.norm(A[i]-x) for i in M]) for j in np.arange(2)])
        its.append(_)
        obj_vals.append(f_d2(x, V, A))
        diff = t.time() - t1
        times.append(diff)
    if _==max_iter-1:
        print("maximum  Weiszfield iterations {} reached".format(max_iter))
    return x,  obj_vals, its, times

def backtrack(x, p, V, A, alpha_0=1, rho=1/2, c=1/4, max_iter=10000):
    """Backtracking line search for gradient descent."""
    alpha = alpha_0
    L = -f_grad(x, V, A).T @ p
    for i in range(max_iter):
        if f_d2(x+alpha*p, V, A) <= f_d2(x, V, A) + c*alpha*L:
            return alpha
            break
        else:
            alpha = rho * alpha
    if i==max_iter-1:
        print("reached maximum backtrack iterations: ", max_iter)
    return alpha

def gradient_descent(V, A, eps=0.001, max_iter=10000):
    """Algorithm using gradient descent instead of the given iteration scheme."""
    x1 = x0 = starting_point(V, A)
    print(x1, x0, "starting point")
    times = []
    obj_vals = []
    t1 = t.time()
    its = []
    for _ in range(max_iter):
        if stop(x0 , V, A, eps):
            break
        p = f_grad(x0, V, A).flatten()
        alpha_k = backtrack(x0, p, V, A)
        x1 = x0 + alpha_k * p
        x0 = x1
        t2 = t.time()-t1
        times.append(t2)
        obj_vals.append(f_d2(x0, V, A))
        its.append(_)
    return x1, obj_vals, its, times

class test1:
    A = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    V = np.ones(len(A))
    
class test2:
    A = np.array([[2, 0], [0, 1], [-1, 0], [0, -1]])
    V = np.array([0.5, 1, 1, 1])
    
class test3:
    A = np.array([[0, 0], [10, 0]])
    V = np.array([5, 1])
    
class test4:
    # p. 18 of lovemorrisbook, gives same result <3
    A = np.array([[1, 1], [1, 4], [2, 2], [4, 5]])
    V = np.array([1, 2, 2, 4])

def update_rc_params():
    fontsize = 21
    newparams = {
        "axes.titlesize": fontsize,
        "axes.labelsize": fontsize,
        "lines.linewidth": 2,
        "lines.markersize": 13,
        "figure.figsize": (13, 7.5),
        "ytick.labelsize": fontsize,
        "xtick.labelsize": fontsize,
        "legend.fontsize": fontsize,
        "legend.handlelength": 1.5,
        "figure.titlesize": fontsize,
        "figure.dpi": 100,
        "text.usetex": True,
        "font.family": "sans-serif",
    }
    plt.rcParams.update(newparams)

def colormesh(test, V, A):
    x = np.linspace(np.min(A[:,0]), np.max(A[:,0]), 50)
    y = np.linspace(np.min(A[:,1]), np.max(A[:,1]), 50)
    X, Y = np.meshgrid(x, y)
    fig = plt.figure()
    for x in X:
        vals =[]
        for y in Y:
            temp = []
            for i in range(len(y)):
                p = np.array([x[i], y[i]])
                temp.append(f_d2(p, V, A))
            vals.append(temp)
    Z = np.array(vals)
    ax = plt.axes()
    im = ax.pcolormesh(X, Y, Z)
    fig.colorbar(im)
    plt.show()

def run_tests(cm=True):
    for test in [test2, test4]:
    
        A = test.A
        V = test.V
        x,y=A.T

        # Compute a reference solution for a small epsilon.
        full_solution, _, _, _ = Weiszfeld(V, A, eps=1e-14)
        reference_solution = f_d2(full_solution, V, A)
        print(reference_solution, full_solution, "REFERENCE")

        # Compute solution with Weiszfeld
        ans_W, f_d2_W, its_W, times_W = Weiszfeld(V, A, eps=1e-8)
        diff_W = np.abs(f_d2_W - reference_solution)

        # Compute solution with gradient.
        ans_G, f_d2_G, its_G, times_G = gradient_descent(V, A, eps=1e-9)
        diff_G = np.abs(f_d2_G - reference_solution)
   
        # Create convergence plot based on iterations.
        fig = plt.figure()
        plt.plot(its_W, diff_W, label="Weiszfeld")
        plt.plot(its_G, diff_G, label="Gradient Descent")
        plt.yscale('log')
        plt.legend()
        plt.xlabel("Iterations")
        plt.ylabel(r"$|f_{d_2}(x)-f_{d_2}(x^*)|$")
        plt.title("Convergence plot")
        plt.show()
        
        # Create convergence plot based on time.
        fig = plt.figure()
        plt.plot(times_W, diff_W, label="Weiszfeld")
        plt.plot(times_G, diff_G,  label="Gradient Descent")
        plt.yscale('log')
        plt.legend()
        plt.xlabel("time [s]")
        plt.ylabel(r"$|f_{d_2}(x)-f_{d_2}(x^*)|$")
        plt.title("Convergence plot")
        plt.show()

        # Visualize solution to problem.
        fig = plt.figure()
        for i in range(len(x)):
            plt.plot(x[i], y[i], ".", label=f"$a^{i + 1} = ({x[i]}, {y[i]}), v^{i + 1}= {V[i]}$")        
        
        cm = True
        plt.plot(ans_G[0], ans_G[1], "g2", label="Gradient-Projected")
        plt.plot(ans_W[0], ans_W[1], "r1", label="Weiszfeld-Projected")
        plt.xlabel(r"$x_1$")
        plt.ylabel(r"$x_2$")
        plt.legend()
        plt.show()
        if cm:
            colormesh(test, V, A)
        
if __name__ == "__main__":
    #update_rc_params() # Remove this if latex is not installed.
    print("starting")
    run_tests()
