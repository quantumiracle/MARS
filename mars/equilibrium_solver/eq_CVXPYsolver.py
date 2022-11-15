""" 
Solve equilibrium (Nash/Correlated) with Linear Programming in zero-sum games. 
Use CVXPY (https://github.com/cvxpy/cvxpy), which can be installed with: pip install cvxpy
"""
import cvxpy as cp
import numpy as np

import time

def NashEquilibriumCVXPYSolver(A, verbose=False):
    rows = A.shape[0]
    cols = A.shape[1]
    x = cp.Variable(rows)
    z = cp.Variable(1)
    multi = np.ones(rows)
    prob = cp.Problem(cp.Maximize(z),
                  [A.T @ x >= z, 0 <= x, x <= 1, multi @ x == 1])
    prob.solve()
    # prob.solve(solver="ECOS")  # may be faster, not significantly. https://github.com/embotech/ecos-python

    p1_value = x.value
    p2_value = prob.constraints[0].dual_value
    
    # There are at least two bad cases with above constrained optimization,
    # where the constraints are not fully satisfied (some numerical issue):
    # 1. the sum of vars is larger than 1.
    # 2. the value of var may be negative.
    abs_p1_value = np.abs(p1_value)
    abs_p2_value = np.abs(p2_value)
    p1_value = abs_p1_value/np.sum(abs_p1_value)
    p2_value = abs_p2_value/np.sum(abs_p2_value)

    if verbose:
        # Print result.
        print("\nThe optimal value is", prob.value)
        print("A solution x is")
        print(p1_value)
        print("A dual solution is")
        print(p2_value)

    return (p1_value, p2_value), None # second should be nash value

if __name__ == "__main__":
    ###   TEST LP NASH SOLVER ###
    A = np.array([[0, -1, 1], [2, 0, -1], [-1, 1, 0]])
    # A=np.array([[ 0.594,  0.554,  0.552,  0.555,  0.567,  0.591],
    # [ 0.575,  0.579,  0.564,  0.568,  0.574,  0.619],
    # [-0.036,  0.28,   0.53,   0.571,  0.57,  -0.292],
    # [ 0.079, -0.141, -0.2,    0.592,  0.525, -0.575],
    # [ 0.545,  0.583,  0.585,  0.562,  0.537,  0.606],
    # [ 0.548,  0.576,  0.58,   0.574,  0.563,  0.564]])

    # A=np.array([[ 0.001,  0.001,  0.00,     0.00,     0.005,  0.01, ],
    # [ 0.033,  0.166,  0.086,  0.002, -0.109,  0.3,  ],
    # [ 0.001,  0.003,  0.023,  0.019, -0.061, -0.131,],
    # [-0.156, -0.039,  0.051,  0.016, -0.028, -0.287,],
    # [ 0.007,  0.029,  0.004,  0.005,  0.003, -0.012],
    # [ 0.014,  0.018, -0.001,  0.008, -0.009,  0.007]])

    t0=time.time()
    ne = NashEquilibriumCVXPYSolver(A, verbose=True)
    print(ne)
    t1=time.time()
    print(t1-t0)