import sys, os
sys.path.append("../..")
from tutorial.solver.common import DoubleOralce, OracleDoubleOralce, QLearningDoubleOralce, FictitiousSelfPlay, OracleFictitiousSelfPlay, QLearningFictitiousSelfPlay
from mars.env.mdp import ArbitraryMDP, MDPWrapper
import numpy as np

def create_test_env():
    num_states = 6
    num_actions_per_player = 6
    num_trans = 6

    env = MDPWrapper(ArbitraryMDP(num_states=num_states, num_actions_per_player=num_actions_per_player, num_trans=num_trans))
    trans_matrices = env.env.trans_prob_matrices # shape: [dim_transition, dim_state, dim_action (p1*p2), dim_state]
    reward_matrices = env.env.reward_matrices # shape: [dim_transition, dim_state, dim_action (p1*p2), dim_state]

    oracle_nash_v, oracle_nash_q, oracle_nash_strategies = env.NEsolver(verbose=False)
    oracle_v_star = oracle_nash_v[0]
    print(np.mean(oracle_v_star, axis=0))
    return env

# solvers = ['DoubleOralce', 'OracleDoubleOralce', 'QLearningDoubleOralce', 'FictitiousSelfPlay', 'OracleFictitiousSelfPlay', 'QLearningFictitiousSelfPlay'][-1]
solvers = ['QLearningDoubleOralce', 'QLearningFictitiousSelfPlay']

env = create_test_env()

# for s in solvers:
#     save_path = f'./data/{s}/'
#     os.makedirs(save_path, exist_ok=True)
#     solver = eval(s)(env, save_path+'data666.npy', solve_episodes=10000)
#     solver.solve()