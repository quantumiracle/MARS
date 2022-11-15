from gurobipy import *
import numpy as np
import itertools
# import gambit
# import gambit.nash
from decimal import *

# Game A represented as array of cost matrices (one per player)
# Cost matrices are multidimensional arrays (as many dimensions as there are players)

def cartesian(arrays, out=None):
    # Returns cartesian product of arrays
    # Props to http://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays/1235363#1235363
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = int(n / arrays[0].size)
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out
    
def arrayWithoutElement(A, i):
    mask = np.ones(A.shape,dtype=bool)
    mask[i]=0
    return A[mask]

def parseGame(A):
    # Returns shape (number of players, number of strategies of P1, number of strategies of P2...)
    # number of players
    # pure moves as tuples of strategies
    shape = np.shape(A)
    num_players = shape[0]
    arrays = []
    for i in range(1, len(shape)):
        arrays.append(range(0, shape[i]))
    pure_moves = cartesian(arrays)
    return (shape, num_players, pure_moves)

def selectMoves(pure_moves, player, strat):
    return [move for move in pure_moves if move[player] == strat]

def getMixedNashEquilibria(A, cost=True):
    # Returns all mixed NE of cost game A (2 players only)
    if cost:
        mat = reversePayoff(A)
    else:
        mat = A
    (shape, num_players, pure_moves) = parseGame(A)
    # g = gambit.new_table(list(shape[1:]))
    g = gambit.Game.new_table(list(shape[1:]))
    for move in pure_moves:
        for player in range(0, num_players):
            g[tuple(move)][player] = Decimal(mat[(player,)+tuple(move)])
    solver = gambit.nash.ExternalEnumMixedSolver()
    a = solver.solve(g)
    ne = []
    for eq in a:
        new_eq = []
        for player in range(0, num_players):
            eq_pl = []
            for strat in range(0, shape[player+1]):
                eq_pl.append(eq[g.players[player].strategies[strat]])
            new_eq.append(eq_pl)
        ne.append(new_eq)
    return ne
    
def getBestAndWorstNE(A, eqs):
    scs = [getSocialCost(A, eq) for eq in eqs]
    return (min(scs), max(scs))
    
def getPureNashEquilibria(A):
    # Needs more testing
    (shape, num_players, pure_moves) = parseGame(A)
    ne = []
    for move in pure_moves:
        is_ne = 1
        for player in range(1, num_players+1):
            payoff = A[player-1,]
            for strat in range(0, shape[player]):
                if strat != move[player-1]:
                    dev = np.copy(move)
                    dev[player-1] = strat
                    if payoff[tuple(dev)] < payoff[tuple(move)]:
                        is_ne = 0
        if is_ne == 1:
            new_ne = []
            for player in range(1, num_players+1):
                ne_str = []
                for strat in range(0, shape[player]):
                    if strat == move[player-1]:
                        ne_str.append(1)
                    else:
                        ne_str.append(0)
                new_ne.append(ne_str)
                        
            ne.append(new_ne)
    print(ne)

def getCorrelatedEquilibria(A, coarse=False, best=True):
    # Works for n players
    A=np.array([A, A])
    (shape, num_players, pure_moves) = parseGame(A)
    p = {}
    m = Model('correlated')
    if best:
        m.ModelSense = 1
    else:
        m.ModelSense = -1
    sums = np.sum(A, axis=0)
    for move in pure_moves:
        s = sums[tuple(move)]
        name = ('p%s' % str(tuple(move)))
        p[tuple(move)] = m.addVar(name=name, obj=s)
        
    m.update()
    m.setParam('OutputFlag', False)
    m.setParam('FeasibilityTol', 1e-9)
    
    printconstr = False
    
    for player in range(0, num_players):
        num_strat = shape[player+1]
        for strat in range(0, num_strat):
            if ~coarse:
                for dev in range(0, num_strat):
                    if strat != dev:
                        responses = selectMoves(pure_moves, player, strat)
                        responses_dev = selectMoves(pure_moves, player, dev)
                        lhs = quicksum(p[tuple(move)]*A[(player,)+tuple(move)] for move in responses)
                        rhs = quicksum(p[tuple(responses[t])]*A[(player,)+tuple(responses_dev[t])] for t in range(0, len(responses)))
                        m.addConstr(lhs <= rhs, name='p'+str(player)+'constr'+str(strat)+'->'+str(dev))
                        if printconstr:
                            print ('--------------------------------------------------------------')
                            print ('Player %d, strat = %d, dev = %d' % (player, strat, dev))
                            print (lhs)
                            print ('<=')
                            print (rhs)
            else:
                responses = selectMoves(pure_moves, player, strat)
                lhs = quicksum(p[tuple(move)]*A[(player,)+tuple(move)] for move in pure_moves)
                rhs = quicksum(quicksum(p[tuple(move)] for move in pure_moves \
                    if (arrayWithoutElement(move, player) == arrayWithoutElement(responses[t], player)).all()) * \
                    A[(player,)+tuple(responses[t])] for t in range(0, len(responses)))
                m.addConstr(lhs <= rhs, name='p'+str(player)+'constr'+str(strat))
                if printconstr:
                    print ('--------------------------------------------------------------')
                    print ('Player %d, strat = %d' % (player, strat))
                    print (lhs)
                    print ('<=')
                    print (rhs)
                    
                        
    m.addConstr(quicksum(p[tuple(move)] for move in pure_moves) == 1, name='proba')
    
    m.optimize()
    
    resp = np.array([v.x for v in m.getVars()])
    resobj = m.objVal
    
    slack = m.getAttr(GRB.attr.Slack, m.getConstrs())
    if np.product([x >= 0 for x in slack]) == 1:
        return (resobj, resp)
    else:
        return (None, None)

def getSocialCost(A, eq):
    (shape, num_players, pure_moves) = parseGame(A)
    sc = 0
    for move in pure_moves:
        mult = 1
        cost = 0
        for player in range(0, num_players):
            mult *= eq[player][move[player]]
            cost += A[(player,)+tuple(move)]
        sc = sc + mult * cost
    return sc

def generateRandomGame(num_players, strategies):
    # num_players is int
    # strategies is list of int representing number of strategies for each player
    return np.random.random_sample((num_players,)+tuple(strategies))

def reversePayoff(A):
    # Gambit oracle computes mixed NE for utilities game, so we reverse the payoffs and feed it in
    # the solver to obtain mixed NE for cost game
    max_payoff = np.amax(A)
    min_payoff = np.amin(A)
    (shape, num_players, pure_moves) = parseGame(A)
    new_A = np.copy(A)
    for move in pure_moves:
        for player in range(0, num_players):
            new_A[(player,)+tuple(move)] = -A[(player,)+tuple(move)]
    return new_A 