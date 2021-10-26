import numpy as np
import numpy as np
import gym
from scipy.sparse import csr_matrix
from .utils.nash_solver import NashEquilibriumECOSSolver

class ArbitraryMDP():
    def __init__(self, num_states=3, num_actions_per_player=2, num_trans=3):
        self.num_states = num_states
        self.num_actions = num_actions_per_player
        self.num_actions_total = self.num_actions**2
        self.observation_space = gym.spaces.Discrete(self.num_states)
        self.action_space = gym.spaces.Discrete(self.num_actions)
        self.max_transition = num_trans
        self.reward_range = [-1,1]
        self.state = None
        self._construct_game()

    def _construct_game(self, ):
        self.trans_prob_matrices, self.reward_matrices = self.generate_random_trans_and_rewards()

    def generate_random_trans_and_rewards(self,):
        trans_prob_matrices = []
        reward_matrices = []
        for _ in range(self.max_transition):
            trans_prob_matrix = []
            reward_matrix = []
            for s in range(self.num_states):
                trans_prob_matrix_for_s = []
                reward_matrix_for_s = []
                for a in range(self.num_actions_total):
                    rands = np.random.uniform(0,1, self.num_states)
                    rand_probs = list(rands/sum(rands))
                    trans_prob_matrix_for_s.append(rand_probs)
                    rs = np.random.uniform(*self.reward_range, self.num_states)
                    reward_matrix_for_s.append(list(rs))
                trans_prob_matrix.append(trans_prob_matrix_for_s)
                reward_matrix.append(reward_matrix_for_s)
            trans_prob_matrices.append(trans_prob_matrix)
            reward_matrices.append(reward_matrix)

        return trans_prob_matrices, reward_matrices

    def reset(self, ):
        self.state = np.random.randint(0, self.num_states)  # randomly pick one state as initial
        self.trans = 0
        obs = self.state
        return obs

    def step(self, a):
        trans_prob = self.trans_prob_matrices[self.trans][self.state][a]
        next_state = np.random.choice([i for i in range(self.num_states)], p=trans_prob)
        self.state = next_state
        obs = self.state
        reward = self.reward_matrices[self.trans][self.state][a][next_state]
        self.trans += 1
        done = False if self.trans < self.max_transition else True

        return obs, reward, done, None

    def NEsolver(self,):
        self.Nash_v = []
        for tm, rm in zip(self.trans_prob_matrices[::-1], self.reward_matrices[::-1]): # inverse enumerate 
            if len(self.Nash_v) > 0:
                rm = np.array(rm)+np.array(self.Nash_v[-1])  # broadcast sum on rm's last dim, last one in Nash_v is for the next state
            trm = np.einsum("ijk,ijk->ij", tm, rm)  # transition prob * reward for the last dimension in (state, action, next_state)
            trm = trm.reshape(-1, self.num_actions, self.num_actions) # action list to matrix
            ne_values = []
            for s_payoff in trm:
                ne = NashEquilibriumECOSSolver(s_payoff)
                ne_value = ne[0]@s_payoff@ne[1].T
                ne_values.append(ne_value)  # each value is a Nash equilibrium value on one state
            self.Nash_v.append(ne_values)  # (trans, state)
        print('Nash values of all states: ', self.Nash_v)

    def action_map(self, action):
        """Action map from one player to two player.
            p2 0  1
        p1
        0      0  1
        1      2  3
        """
        a = action[0]*self.num_actions+action[1]
        return a

if __name__ == '__main__':
    from mdp_wrapper import MDPWrapper

    # single agent version
    # env = ArbitraryMDP()
    # obs = env.reset()
    # print(obs)
    # done = False
    # while not done:
    #     obs, r, done, _ = env.step(0)
    #     print(obs, r, done)

    # two agent version
    env = MDPWrapper(ArbitraryMDP())
    print(env.observation_space, env.action_space)
    # env.render()
    obs = env.reset()
    print(obs)
    done = False
    while not np.any(done):
        obs, r, done, _ = env.step([0,1])
        print(obs, r, done)
