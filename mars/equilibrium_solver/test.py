# for python 3 testing compatibility
# ref: https://github.com/embotech/ecos-python/pull/20

from __future__ import print_function
import platform
from multiprocessing.pool import ThreadPool

def import_error(msg):
    print()
    print("## IMPORT ERROR:", msg)
    print()

try:
    from nose.tools import assert_raises, assert_almost_equals
except ImportError:
    import_error("Please install nose to run tests.")
    raise

try:
    import ecos
except ImportError:
    import_error("You must install the ecos module before running tests.")
    raise

try:
    import numpy as np
except ImportError:
    import_error("Please install numpy.")
    raise

try:
    import scipy.sparse as sp
except ImportError:
    import_error("Please install scipy.")
    raise

# global data structures for problem
n = 100
m = 50
p = 50
c = np.random.randn(n)
h = np.random.randn(p)
G = sp.csc_matrix(np.random.randn(p, n))
A = sp.csc_matrix(np.random.randn(m, n))
b = np.random.randn(m)
dims = {'q': [], 'l': p}

myopts = {'feastol': 2e-8, 'reltol': 2e-8, 'abstol': 2e-8, 'verbose': False}


def f(i):
    sol = ecos.solve(c, G, h, dims, **myopts)
    return sol

import time

pool = ThreadPool(8)
tic = time.time()
results = pool.map(f, range(3))
print(results)
toc = time.time()

print(toc - tic)