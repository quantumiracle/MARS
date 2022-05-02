import numpy as np
import math
import gym
from gym.spaces import Box
import scipy.linalg

from .utils.nash_solver import NashEquilibriumECOSSolver
# from utils.nash_solver import NashEquilibriumECOSSolver


class RichObsArbitraryMDP():
    def __init__(self, num_states=3, num_actions_per_player=3, num_trans=2, given_trans=None, given_rewards=None):
        self.num_states = num_states  # number of states for each timestep
        self.num_actions = num_actions_per_player
        self.num_actions_total = self.num_actions**2
        #self.observation_space = gym.spaces.Discrete(self.num_states*(num_trans+1))

        self.observation_dim = 2 ** int(math.ceil(np.log2(num_trans+num_states+1)))
        self.observation_space = Box(low=0.0, high=1.0, shape=(self.observation_dim,),dtype=np.float)

        self.action_space = gym.spaces.Discrete(self.num_actions)
        self.max_transition = num_trans
        self.reward_range = [-1,1]
        self.state = None
        self.given_trans = given_trans
        self.given_rewards = given_rewards
        self.OneHotObs = False
        self.noise = 0.1

        self.rotation = scipy.linalg.hadamard(self.observation_space.shape[0])
        ## rock-paper-scissor test
        # self.given_rewards = [
        #             [[ [0], [-1], [1],
        #                 [1], [0], [-1],
        #                 [-1], [1], [0],]], 
        #                 ]

        ## A fixed simple test, with: num_states=1, num_actions_per_player=3, num_trans=2
        # self.given_trans = [
        #             [[ [0.5, 0.5], [0.5, 0.5], [0.5, 0.5],
        #                 [0.5, 0.5], [0.5, 0.5], [0.5, 0.5],
        #                 [0.5, 0.5], [0.5, 0.5], [0.5, 0.5],],

        #                [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5],
        #                 [0.5, 0.5], [0.5, 0.5], [0.5, 0.5],
        #                 [0.5, 0.5], [0.5, 0.5], [0.5, 0.5],]] ,

        #              [[ [0.5, 0.5], [0.5, 0.5], [0.5, 0.5],
        #                 [0.5, 0.5], [0.5, 0.5], [0.5, 0.5],
        #                 [0.5, 0.5], [0.5, 0.5], [0.5, 0.5],],

        #                [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5],
        #                 [0.5, 0.5], [0.5, 0.5], [0.5, 0.5],
        #                 [0.5, 0.5], [0.5, 0.5], [0.5, 0.5],]] 
        #                 ]

                    # [[ [1, 0], [1, 0], [1, 0],
                    #         [1, 0], [1, 0], [1, 0],
                    #         [1, 0], [1, 0], [1, 0],],

                    #     [[1, 0], [1, 0], [1, 0],
                    #         [1, 0], [1, 0], [1, 0],
                    #         [1, 0], [1, 0], [1, 0],]] ,

                    #     [[ [1, 0], [1, 0], [1, 0],
                    #         [1, 0], [1, 0], [1, 0],
                    #         [1, 0], [1, 0], [1, 0],],

                    #     [[1, 0], [1, 0], [1, 0],
                    #         [1, 0], [1, 0], [1, 0],
                    #         [1, 0], [1, 0], [1, 0],]] 
                    #         ]

        # self.given_rewards = [
        #             [[ [0, 0], [2, 2], [-1, -1],
        #                 [-1, -1], [0,0], [1,1],
        #                 [1,1], [-1,-1], [0,0],],

        #                [[0, 0], [2, 2], [-1, -1],
        #                 [-1, -1], [0,0], [1,1],
        #                 [1,1], [-1,-1], [0,0],]] ,

        #              [[ [0, 0], [2, -2], [-1, -1],
        #                 [-1, -1], [0,0], [1,1],
        #                 [1,1], [-1,-1], [0,0],],

        #                [[0, 0], [2, -2], [-1, -1],
        #                 [-1, -1], [0,0], [1,1],
        #                 [1,1], [-1,-1], [0,0],]] 
        #                 ]

        self.seed(0)  # if want random game, uncomment this line
        self._construct_game()
        self.NEsolver()

    def _construct_game(self, ):
        # shape: [dim_transition, dim_state, dim_action (p1*p2), dim_state]
        self.trans_prob_matrices, self.reward_matrices = self.generate_random_trans_and_rewards()
        # print(self.trans_prob_matrices, self.reward_matrices)

    def seed(self, seed):
        # this seed is actually not userfull since it's after game construction
        np.random.seed(seed)

    def generate_random_trans_and_rewards(self, SameRewardForNextState=False):
        """Generate arbitrary transition matrix and reward matrix.
        :param SameRewardForNextState: r(s,a) if True else r(s,a,s')
        :type SameRewardForNextState: bool
        :return: the list of transition matrix and the list of reward matrix, 
        both in shape: (dim_transition, dim_state, dim_action, dim_state)
        :rtype: [type]
        """
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
                    if SameRewardForNextState:  # r(s,a) this reduces stochasticity in nash value estimation thus work!
                        rs = int(self.num_states) * [np.random.uniform(*self.reward_range)]
                        reward_matrix_for_s.append(rs)
                    else:  # r(s,a,s')
                        rs = np.random.uniform(*self.reward_range, self.num_states)
                        reward_matrix_for_s.append(list(rs))
 
                trans_prob_matrix.append(trans_prob_matrix_for_s)
                reward_matrix.append(reward_matrix_for_s)
            trans_prob_matrices.append(trans_prob_matrix)
            reward_matrices.append(reward_matrix)

        if self.given_trans is not None:
            trans_prob_matrices = self.given_trans
        if self.given_rewards is not None:
            reward_matrices = self.given_rewards

        return trans_prob_matrices, reward_matrices

    def reset(self, ):
        self.state = np.random.randint(0, self.num_states)  # randomly pick one state as initial
        self.trans = 0

        obs = self.make_obs()

        return obs
    # def step(self, a):
    #     """The environment transition function.
    #     For a given state and action, the transition is stochastic. 
    #     For representation of states, considering the num_states=3 case, the first three states are (0,1,2);
    #     after one transition, the possible states are (3,4,5), etc. Such that states after different numbers of 
    #     transitions can be distinguished. 
        
    #     :param a: action
    #     """
    #     trans_prob = self.trans_prob_matrices[self.trans][self.state%self.num_states][a]
    #     next_state = np.random.choice([i for i in range(self.num_states)], p=trans_prob) + (self.trans+1) * self.num_states
    #     reward = self.reward_matrices[self.trans][self.state%self.num_states][a][next_state%self.num_states]

    #     self.state = next_state
    #     obs = self.state
    #     self.trans += 1
    #     done = False if self.trans < self.max_transition else True
    #     if self.OneHotObs:
    #         return self._to_one_hot(obs), reward, done, None
    #     else:
    #         return obs, reward, done, None

    def step(self, a, s=None):
        """The environment transition function.
        For a given state and action, the transition is stochastic. 
        For representation of states, considering the num_states=3 case, the first three states are (0,1,2);
        after one transition, the possible states are (3,4,5), etc. Such that states after different numbers of 
        transitions can be distinguished. 
        
        :param a: action
        """
        if s is not None:
            # set a state s for debugging
            self.trans = int(s/self.num_states)
            self.state = s
            
        trans_prob = self.trans_prob_matrices[self.trans][self.state%self.num_states][a]
        next_state = np.random.choice([i for i in range(self.num_states)], p=trans_prob) + (self.trans+1) * self.num_states
        reward = self.reward_matrices[self.trans][self.state%self.num_states][a][next_state%self.num_states]

        self.state = next_state
        #obs = self.state
        self.trans += 1
        done = False if self.trans < self.max_transition else True

        obs = self.make_obs()

        return obs, reward, done, None

    def _to_one_hot(self, s):
        one_hot_vec = np.zeros(self.num_states*(self.max_transition+1))
        one_hot_vec[s] = 1
        return one_hot_vec

    def make_obs(self):

        gaussian = np.zeros(self.observation_space.shape)
        gaussian[:(self.max_transition+self.num_states+1)] = np.random.normal(0,self.noise,[self.max_transition+self.num_states+1]) # self.trans can be self.max_transition+1
        gaussian[self.state%self.num_states] += 1  # self.state%self.num_states gives the ture state index at current step
        gaussian[self.num_states+self.trans] += 1

        x = (self.rotation*np.matrix(gaussian).T).T
        return np.reshape(np.array(x), x.shape[1])

    def NEsolver(self, verbose = False):
        """
        Formulas for calculating Nash equilibrium strategies and values:
        1. Nash strategies: (\pi_a^*, \pi_b^*) = \min \max Q(s,a,b), 
            where Q(s,a,b) = r(s,a,b) + \gamma \min \max Q(s',a',b') (this is the definition of Nash Q-value);
        2. Nash value: Nash V(s) = \min \max Q(s,a,b) = \pi_a^* Q(s,a,b) \pi_b^{*T}
        """

        self.Nash_v = []
        self.Nash_q = []
        self.Nash_strategies = []
        for tm, rm in zip(self.trans_prob_matrices[::-1], self.reward_matrices[::-1]): # inverse enumerate 
            if len(self.Nash_v) > 0:
                rm = np.array(rm)+np.array(self.Nash_v[-1])  # broadcast sum on rm's last dim, last one in Nash_v is for the next state
            nash_q_values = np.einsum("ijk,ijk->ij", tm, rm)  # transition prob * reward for the last dimension in (state, action, next_state)
            nash_q_values = nash_q_values.reshape(-1, self.num_actions, self.num_actions) # action list to matrix
            self.Nash_q.append(nash_q_values)
            ne_values = []
            ne_strategies = []
            for nash_q_value in nash_q_values:
                ne = NashEquilibriumECOSSolver(nash_q_value)
                ne_strategies.append(ne)
                ne_value = ne[0]@nash_q_value@ne[1].T
                ne_values.append(ne_value)  # each value is a Nash equilibrium value on one state
            self.Nash_v.append(ne_values)  # (trans, state)
            self.Nash_strategies.append(ne_strategies)
        self.Nash_v = self.Nash_v[::-1]
        self.Nash_q = self.Nash_q[::-1]  # (dim_transition, dim_state, dim_action, dim_action)
        self.Nash_strategies = self.Nash_strategies[::-1]  # (dim_transition, dim_state, #players, dim_action)
        if verbose:
            print('Nash values of all states (from start to end): \n', self.Nash_v)
            print('Nash Q-values of all states (from start to end): \n', self.Nash_q)
            print('Nash strategies of all states (from start to end): \n', self.Nash_strategies)

        ## To evaluate the correctness of the above values
        # for v, q, s in zip(self.Nash_v, self.Nash_q, self.Nash_strategies):
        #     for vv,qq,ss in zip(v,q,s):
        #         cal_v = ss[0]@qq@ss[1].T
        #         print(vv, cal_v)

        return self.Nash_v, self.Nash_q, self.Nash_strategies

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
    env = MDPWrapper(RichObsArbitraryMDP())
    nash_v, _, nash_strategies = env.NEsolver()
    print(nash_strategies, np.array(nash_strategies).shape)
    # np.save('../../../data/nash_dqn_test/oracle_nash.npy', nash_strategies)
    print('oracle nash v star: ', np.mean(nash_v[0], axis=0))  # the average nash value for initial states from max-player's view
    print('spaces: ', env.observation_space, env.action_space)
    # env.render()
    obs = env.reset()
    print(obs)
    done = False
    while not np.any(done):
        obs, r, done, _ = env.step([1,0])
        print(obs, r, done)
