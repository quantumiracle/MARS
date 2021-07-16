import numpy as np
from rl.algorithm.equilibrium_solver import *

################### TEST #################
A = np.array([[2, 0], [6, 5]])
B = np.array([[2, 6], [0, 5]])
nes = NashEquilibriaSolver(A, B)
print(nes)

A = np.array([[10, 0], [6, 2]])
B = np.array([[9, 6], [0, 2]])
nes = NashEquilibriaSolver(A, B)
print(nes)
    
A = np.array([[3, -1], [-1, 1]])
nes = NashEquilibriaSolver(A)
print(nes)