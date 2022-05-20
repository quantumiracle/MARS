import torch
import numpy as np
import pickle
from mars.equilibrium_solver import NashEquilibriumECOSSolver, NashEquilibriumMWUSolver, NashEquilibriumParallelMWUSolver

DEBUG = False

def kl(p, q):
    """Kullback-Leibler divergence D(P || Q) for discrete distributions
    Parameters
    ----------
    p, q : array-like, dtype=float, shape=n
    Discrete probability distributions.
    """
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)

    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def to_one_hot(s, range):
    one_hot_vec = np.zeros(range)
    one_hot_vec[s] = 1
    return one_hot_vec

class Debugger():
    def __init__(self, env, log_path = None):
        self.env = env
        if env.OneHotObs:
            self.num_states_per_step = int(self.env.observation_space.shape[0])
        else:
            self.num_states_per_step = int(self.env.observation_space.high[0]/(self.env.max_transition+1))
        self.max_transition = env.max_transition
        self.kl_dist_list=[[] for _ in range(self.max_transition)]
        self.mse_v_list=[[] for _ in range(self.max_transition)]
        self.mse_exp_list=[[] for _ in range(self.max_transition)]
        self.brv_list = []
        self.cnt = 0
        self.save_interval = 10
        self.logging = {'num_states_per_step': self.num_states_per_step,
                        'max_transition': self.max_transition,
                        'oracle_exploitability': np.mean(self.env.Nash_v[0], axis=0),  # the average nash value for initial states from max-player's view
                        'cnt': [],
                        'state_visit': {},
                        'kl_nash_dist': [],
                        'mse_nash_v': [],
                        'mse_exploitability': []
                        }
        self.log_path = log_path 
        self.state_list = []

        self.oracle_nash_strategies = np.vstack(self.env.Nash_strategies) # flatten to shape dim 1
        self.oracle_nash_values = np.concatenate(self.env.Nash_v) # flatten to shape dim 1
        self.oracle_nash_q_values = np.concatenate(self.env.Nash_q) # flatten to shape dim 1
        self.trans_prob_matrices = self.env.env.trans_prob_matrices
        self.reward_matrices = self.env.env.reward_matrices
        print('oracle nash v star: ', np.mean(self.env.Nash_v[0], axis=0))  # the average nash value for initial states from max-player's view

    def best_response_value(self, learned_q):
        """
        Formulas for calculating best response values:
        1. Nash strategies: (\pi_a^*, \pi_b^*) = \min \max Q(s,a,b), 
            where Q(s,a,b) = r(s,a,b) + \gamma \min \max Q(s',a',b') (this is the definition of Nash Q-value);
        2. Best response (of max player) value: Br V(s) = \min_b \pi(s,a) Q(s,a,b)
        """
        Br_v = []
        Br_q = []
        Nash_strategies = []
        num_actions = learned_q.shape[-1]
        for tm, rm, qm in zip(self.trans_prob_matrices[::-1], self.reward_matrices[::-1], learned_q[::-1]): # inverse enumerate 
            if len(Br_v) > 0:
                rm = np.array(rm)+np.array(Br_v[-1])  # broadcast sum on rm's last dim, last one in Nash_v is for the next state
            br_q_values = np.einsum("ijk,ijk->ij", tm, rm)  # transition prob * reward for the last dimension in (state, action, next_state)
            br_q_values = br_q_values.reshape(-1, num_actions, num_actions) # action list to matrix
            Br_q.append(br_q_values)
            br_values = []
            ne_strategies = []
            for q, br_q in zip(qm, br_q_values):
                ne, _ = NashEquilibriumECOSSolver(q)
                ne_strategies.append(ne)
                br_value = np.min(ne[0]@br_q)  # best response againt "Nash" strategy of first player
                br_values.append(br_value)  # each value is a Nash equilibrium value on one state
            Br_v.append(br_values)  # (trans, state)
            Nash_strategies.append(ne_strategies)
        Br_v = Br_v[::-1]  # (#trans, #states)
        Br_q = Br_q[::-1]
        Nash_strategies = Nash_strategies[::-1]

        avg_init_br_v = -np.mean(Br_v[0])  # average best response value of initial states; minus for making it positive
        return avg_init_br_v

    def compare_with_oracle(self, state, dists, ne_vs, ne_q_vs, verbose=False):
        """[summary]

        :param state: current state
        :type state: [type]
        :param dists: predicted Nash strategies (distributions)
        :type dists: [type]
        :param ne_vs: predicted Nash equilibrium values based on predicted Nash strategies
        :type ne_vs: [type]
        :param verbose: [description], defaults to False
        :type verbose: bool, optional
        """
        self.cnt+=1
        if self.env.OneHotObs:
            state_ = state[0].cpu().numpy()
            id_state = np.where(state_>0)[0][0]
        else:
            id_state =  int(torch.sum(state).cpu().numpy()/2)

        for j in range(self.max_transition):  # nash value for non-terminal states (before the final timestep)
            if id_state >= j*self.num_states_per_step and id_state < (j+1)*self.num_states_per_step:  # determine which timestep is current state
                ne_strategy = self.oracle_nash_strategies[id_state]
                ne_v = self.oracle_nash_values[id_state]
                ne_q = self.oracle_nash_q_values[id_state]
                oracle_first_player_ne_strategy = ne_strategy[0]
                nash_dqn_first_player_ne_strategy = dists[0][0]
                br_v = np.min(nash_dqn_first_player_ne_strategy@ne_q)  # best response value (value against best response), reflects exploitability of learned Nash; but this minimization is taken with oracle nash 
                kl_dist = kl(oracle_first_player_ne_strategy, nash_dqn_first_player_ne_strategy)
                self.kl_dist_list[j].append(kl_dist)
                mse_v = float((ne_v - ne_vs)**2) # squared error of Nash values (predicted and oracle)
                self.mse_v_list[j].append(mse_v)
                ### this is the exploitability/regret for each state; but not calcuated correctly, the minimization should take over best-response Q value rather than nash Q (neither oracle nor learned)
                mse_exp = float((ne_v - br_v)**2)  # the target value of best response value (exploitability) should be the Nash value
                self.mse_exp_list[j].append(mse_exp)

        ## this is the correct calculation of exploitability: average best-response value of the inital states
        brv = self.best_response_value(ne_q_vs, )
        self.brv_list.append(brv)

        self.state_visit(id_state)
        self.log([id_state, kl_dist, ne_vs], verbose)
        if self.cnt % self.save_interval == 0:
            self.dump_log()

    def state_visit(self, state):
        self.state_list.append(state)


    def log(self, data, verbose=False):
        # get state visitation statistics
        unique, counts = np.unique(self.state_list, return_counts=True)
        state_stat = dict(zip(unique, counts))
        if verbose:
            print('state index: {}， KL: {}'.format(*data))
            print('state visitation counts: {}'.format(state_stat))

        self.logging['cnt'].append(self.cnt)
        self.logging['state_visit'] = state_stat
        self.logging['kl_nash_dist'] = self.kl_dist_list
        self.logging['mse_nash_v'] = self.mse_v_list
        self.logging['mse_exploitability'] = self.mse_exp_list
        self.logging['brv'] = self.brv_list

    def dump_log(self,):
        with open(self.log_path, "wb") as f:
            pickle.dump(self.logging, f)    

