""" Solve equilibrium (Nash/Correlated) with Linear Programming in zero-sum games. """
import numpy as np
import gurobipy as gp
import time

def NashEquilibriumGUROBISolver(game_matrix):
    '''
    Finds a minimax equilibrium of the given zero-sum game via linear programming. Returns the mixed strategy
    of the row and column player respectively.
    
    Assumes that the row player wants to maximize their payoff
    '''
    import gurobipy as gp
    game_matrix = np.array(game_matrix)
    m = gp.Model('adversary_model')
    m.params.OutputFlag = 0
    row_vars = []
    #variables giving the probability that each strategy is played by the row player
    for r in range(game_matrix.shape[0]):
        row_vars.append(m.addVar(vtype = gp.GRB.CONTINUOUS, name = 'row_' + str(r), lb = 0, ub = 1))
    #the value of the game
    v = m.addVar(vtype = gp.GRB.CONTINUOUS, name = 'value')
    m.update()
    #constraint that the probabilities sum to 1
    m.addConstr(gp.quicksum(row_vars) == 1, 'row_normalized')
    #constrain v to the be the minimum expected reward over all pure column strategies
    column_constraints = []
    for i in range(game_matrix.shape[1]):
        column_constraints.append(m.addConstr(v <= gp.quicksum(game_matrix[j, i]*row_vars[j] for j in range(len(row_vars))), 'response_' + str(i)))
    #objective is to maximize the value of the game
    m.setObjective(v, gp.GRB.MAXIMIZE)
    #solve
    m.optimize()        
    #get the row player's mixed strategy
    row_mixed = []
    for r in range(game_matrix.shape[0]):
        row_mixed.append(row_vars[r].x)
    #get the column player's mixed strategy from the dual variable associated with each constraint
    column_mixed = []
    for c in range(game_matrix.shape[1]):
        column_mixed.append(column_constraints[c].getAttr('Pi'))
    return (row_mixed, column_mixed)

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
    ne = NashEquilibriumGUROBISolver(A)
    print(ne)
    t1=time.time()
    print(t1-t0)