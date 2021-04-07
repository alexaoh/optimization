from cg_method import *
from scipy.linalg import hilbert

# Exercise set 5, problem 2.
# Solve systems where A is the Hilbert matrix of the given matrix and rhs is b = (1,1,...,1)^T. x0 = 0.
# Compare the amount of iterations to reach residual < 10e-6 with the theoretical results for the CG method. 
dimensions = [5, 8, 12, 20]
iterations = [0, 0, 0, 0]
tol = 1e-6
for i, d in enumerate(dimensions):
    x0 = np.zeros(d)
    b = np.ones(d)
    A = hilbert(d)
    _, it, r = cg(x0, A, b, tol)
    iterations[i] = it

print("Iterations required to reduce the \ell^2 norm below", str(tol) + ", for the different dimensions are")
print(iterations)

# Why do these results not contradict the theoretical results concerning the CG method that were discussed in the lecture?
# As is seen by the results, the number of iterations are always larger than the number of dimensions. 
# This contradicts the theoretical results, which state that convergence always takes place in mostly the same amount of iterations
# as the number of variables. However, in this theoretical results, it is assumed that the arithmetic operations are exact, 
# which is not the case when applied numerically. 

# The expected effect of the rounding errors depends ont he condition number of the matrix A. 
# Since the Hilbert matrix is badly conditioned ( \mathcal{O}(\frac{(1+\sqrt{2})^{4n}}{\sqrt{n}}) ), 
# the condition number increases very fast, and it is expected that the CG methods struggles for this problem. 
