""" Solve equilibrium (Nash/Correlated) with Linear Programming in zero-sum games for non-square matrix. """
import ecos
import numpy as np
from scipy.sparse import csr_matrix
import time

def NashEquilibriumECOSSolver(M):
    """
    https://github.com/embotech/ecos-python
    min  c*x
    s.t. A*x = b
         G*x <= h
    https://github.com/embotech/ecos/wiki/Usage-from-MATLAB
    args:
        c,b,h: numpy.array
        A, G: Scipy sparse matrix
    """
    row, col = M.shape
    c = np.zeros(row+1)
    # max z
    c[-1] = -1

    # x1+x2+...+xn=1
    A = np.ones(row+1)
    A[-1] = 0.
    A = csr_matrix([A])
    b=np.array([1.])

    # M.T*x<=z
    G1 = np.ones((col, row+1))
    G1[:col, :row] = -1. * M.T
    # x>=0
    G2 = np.zeros((row, row+1))
    for i in range(row):
        G2[i, i]=-1.
    # x<=1.
    G3 = np.zeros((row, row+1))
    for i in range(row):
        G3[i, i]=1.
    G = csr_matrix(np.concatenate((G1, G2, G3)))
    # h = np.concatenate((np.zeros(2*row), np.ones(row)))
    h = np.concatenate((np.zeros(col), np.zeros(row), np.ones(row)))

    # specify number of variables
    dims={'l': col+2*row, 'q': []}

    solution = ecos.solve(c,G,h,dims,A,b, verbose=False)

    p1_dist = solution['x'][:row]
    p2_dist = solution['z'][:col] # z is the dual variable of x
    # There are at least two bad cases with above constrained optimization,
    # where the constraints are not fully satisfied (some numerical issue):
    # 1. the sum of vars is larger than 1.
    # 2. the value of var may be negative.
    abs_p1_dist = np.abs(p1_dist)
    abs_p2_dist = np.abs(p2_dist)
    normalized_p1_dist = abs_p1_dist/np.sum(abs_p1_dist)
    normalized_p2_dist = abs_p2_dist/np.sum(abs_p2_dist)

    nash_value = normalized_p1_dist@M@normalized_p2_dist.T

    return (normalized_p1_dist, normalized_p2_dist), nash_value



if __name__ == "__main__":
    A = np.array([[0, -1, ], [1, 0, ], [-1, 1, ]])  # RPS vs RP
    ne, ne_v = NashEquilibriumECOSSolver(A)
    print(ne, ne_v)