import sys, os
sys.path.append("../..")
from tutorial.solver.common import DoubleOralce, OracleDoubleOralce, FictitiousSelfPlay, OracleFictitiousSelfPlay
from mars.env.mdp import ArbitraryMDP, MDPWrapper
import numpy as np

def create_test_env():
    num_states = 3
    num_actions_per_player = 3
    num_trans = 3

    env = MDPWrapper(ArbitraryMDP(num_states=num_states, num_actions_per_player=num_actions_per_player, num_trans=num_trans))
    trans_matrices = env.env.trans_prob_matrices # shape: [dim_transition, dim_state, dim_action (p1*p2), dim_state]
    reward_matrices = env.env.reward_matrices # shape: [dim_transition, dim_state, dim_action (p1*p2), dim_state]

    oracle_nash_v, oracle_nash_q, oracle_nash_strategies = env.NEsolver(verbose=False)
    oracle_v_star = oracle_nash_v[0]

    return env

solvers = ['DoubleOralce', 'OracleDoubleOralce', 'FictitiousSelfPlay', 'OracleFictitiousSelfPlay'][-1]

env = create_test_env()
save_path = f'./data/{solvers}/'
os.makedirs(save_path, exist_ok=True)
solver = eval(solvers)(env, save_path+'data.npy', solve_episodes=1000)
solver.solve()