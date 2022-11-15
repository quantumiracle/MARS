### Test Nash equilibrium solvers ###

import sys
sys.path.append("..")

import numpy as np
import time
from mars.equilibrium_solver import *


Solvers = ['NashEquilibriaSolver', 'NashEquilibriumSolver', 
        'NashEquilibriumLPSolver', 'NashEquilibriumCVXPYSolver', 
        'NashEquilibriumGUROBISolver', 'NashEquilibriumECOSSolver', 
        'NashEquilibriumMWUSolver'][-2:]

Matrix_size = 6
Test_times = 100
################### TEST #################
random_matrices = [np.random.uniform(-1,1,Matrix_size**2).reshape(Matrix_size, Matrix_size) for _ in range(Test_times)]

for solver in Solvers:
    t0=time.time()
    for i in range(Test_times):
        ne, nev = eval(solver)(random_matrices[i])

    t1=time.time()
    print(solver, ',time taken per matrix: ', (t1-t0)/Test_times)