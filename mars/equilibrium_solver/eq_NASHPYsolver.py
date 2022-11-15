import nashpy as nash
import numpy as np
from .gamegenerator import getCorrelatedEquilibria, getMixedNashEquilibria, getPureNashEquilibria 
from .lamke_howson_lex import lemke_howson_lex 

def NashEquilibriaSolver(A,B=None):
    """
    Given payoff matrix/matrices, return a list of existing Nash equilibria:
    [(nash1_p1, nash1_p2), (nash2_p1, nash2_p2), ...]
    """
    if B is not None:
        rps = nash.Game(A, B)
    else:
        rps = nash.Game(A  )# zero-sum game: unimatrix  
    # eqs = rps.support_enumeration()
    eqs = rps.vertex_enumeration()
    return list(eqs)

def NashEquilibriumSolver(A, B=None):
    """
    Quickly solve *one* Nash equilibrium with Lemke Howson algorithm, given *degenerate* (det not equal to 0) payoff matrix.
    Ref: https://nashpy.readthedocs.io/en/stable/reference/lemke-howson.html#lemke-howson

    TODO: sometimes give nan or wrong dimensions, check here: https://github.com/drvinceknight/Nashpy/issues/35
    """
    # print('Determinant of matrix: ', np.linalg.det(A))
    if B is not None:
        rps = nash.Game(A, B)
    else:
        rps = nash.Game(A)  # zero-sum game: unimatrix  
    dim = A.shape[0]
    final_eq = None
    # To handle the problem that sometimes Lemke-Howson implementation will give 
    # wrong returned NE shapes or NAN in value, use different initial_dropped_label value 
    # to find a valid one. 
    for l in range(0, sum(A.shape) - 1):
        # Lemke Howson can not solve degenerate matrix.
        # eq = rps.lemke_howson(initial_dropped_label=l) # The initial_dropped_label is an integer between 0 and sum(A.shape) - 1

        # Lexicographic Lemke Howson can solve degenerate matrix: https://github.com/newaijj/Nashpy/blob/ffea3522706ad51f712d42023d41683c8fa740e6/tests/unit/test_lemke_howson_lex.py#L9 
        eq = lemke_howson_lex(A, -A, initial_dropped_label=l)  
        if eq[0].shape[0] ==  dim and eq[1].shape[0] == dim and not np.isnan(eq[0]).any() and not np.isnan(eq[1]).any():
            # valid shape and valid value (not nan)
            final_eq = eq
            break
    if final_eq is None:
        raise ValueError('No valid Nash equilibrium is found!')
    return final_eq

if __name__ == "__main__":
    # A = np.array([[0, -1, 1], [2, 0, -1], [-1, 1, 0]])
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

    A = np.array([[3, -1], [-1, 1]])

    import time

    t0 = time.time()
    nes = NashEquilibriaSolver(A)
    print(nes)
    t1 = time.time()
    print(t1-t0)

    ne = NashEquilibriumSolver(A)
    t2 = time.time()
    print(ne)
    print(t2-t1)

    # cce = getCorrelatedEquilibria(A, coarse=False)
    # print(cce)

    # ne = getMixedNashEquilibria(A)
    # print(ne)

