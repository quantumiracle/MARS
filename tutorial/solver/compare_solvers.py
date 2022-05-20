import sys, os
sys.path.append("../..")
from tutorial.solver.common import SelfPlay, QLearningSelfPlay, QLearningFictitiousSelfPlay,\
     QLearningDoubleOralce, OracleDoubleOralce, FictitiousSelfPlay, OracleFictitiousSelfPlay
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
    oracle_v_star = np.mean(oracle_nash_v[0])
    print(np.mean(oracle_v_star))

    return env

solvers = ['QLearningSelfPlay', 'QLearningFictitiousSelfPlay', 'QLearningDoubleOralce',]
episodes = len(solvers)*[10000]
env = create_test_env()

for solver, epi in zip(solvers, episodes):
    print(solver, epi)
    save_path = f'./data/{solver}/'
    os.makedirs(save_path, exist_ok=True)
    s = eval(solver)(env, save_path+'data333.npy', solve_episodes=epi)
    s.solve()