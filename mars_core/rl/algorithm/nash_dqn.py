import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import gym
import operator
import random, copy
from .common.nn_components import cReLU, Flatten
from .common.storage import ReplayBuffer
from .common.rl_utils import choose_optimizer, EpsilonScheduler
from .common.networks import NetBase, get_model
from .dqn import DQN, DQNBase
from .equilibrium_solver import NashEquilibriumECOSSolver

DEBUG = True

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
class NashDQN(DQN):
    """
    Nash-DQN algorithm
    """
    def __init__(self, env, args):
        super().__init__(env, args)
        self.num_envs = args.num_envs
        self.model = NashDQNBase(env, args.net_architecture, args.num_envs).to(self.device)
        self.target = copy.deepcopy(self.model).to(self.device)
        self.num_agents = env.num_agents[0] if isinstance(env.num_agents, list) else env.num_agents
        try:
            self.action_dims = env.action_space[0].n
        except:
            self.action_dims = env.action_space.n
        # don't forget to instantiate an optimizer although there is one in DQN
        self.optimizer = choose_optimizer(args.optimizer)(self.model.parameters(), lr=float(args.learning_rate))

        if DEBUG:
            self.env = env
            self.kl_list=[[] for _ in range(3)]

    def choose_action(self, state, Greedy=False, epsilon=None):
        if Greedy:
            epsilon = 0.
        elif epsilon is None:
            epsilon = self.epsilon_scheduler.get_epsilon()
        if not isinstance(state, torch.Tensor):
            state = torch.Tensor(state).to(self.device)
        if self.num_envs == 1: # state: (agents, state_dim)
            state = state.unsqueeze(0).view(1, -1) # change state from (agents, state_dim) to (1, agents*state_dim)
        else: # state: (agents, envs, state_dim)
            state = torch.transpose(state, 0, 1) # to state: (envs, agents, state_dim)
            state = state.view(state.shape[0], -1) # to state: (envs, agents*state_dim)

        if random.random() > epsilon:  # NoisyNet does not use e-greedy
            with torch.no_grad():
                q_values = self.model(state).detach().cpu().numpy()  # needs state: (batch, agents*state_dim)
            # if self.args.cce:
            #     actions = self.compute_cce(q_values)
            # else:
            actions, dists = self.compute_nash(q_values) 

            if DEBUG: ## test on arbitrary MDP
                id_state =  int(torch.sum(state).cpu().numpy()/2)
                nash_strategies = np.vstack(self.env.Nash_strategies)
                if id_state < 3: 
                    # print(id_state)
                    # print(self.env.Nash_strategies[0][id_state])
                    ne_strategy = nash_strategies[id_state]
                    oracle_first_player_ne_strategy = ne_strategy[0]
                    nash_dqn_first_player_ne_strategy = dists[0][0]
                    # print(first_player_ne_strategy, nash_dqn_first_player_ne_strategy)
                    kl_dist = kl(oracle_first_player_ne_strategy, nash_dqn_first_player_ne_strategy)
                    self.kl_list[0].append(kl_dist)

                elif id_state < 6:
                    ne_strategy = nash_strategies[id_state]
                    oracle_first_player_ne_strategy = ne_strategy[0]
                    nash_dqn_first_player_ne_strategy = dists[0][0]
                    kl_dist = kl(oracle_first_player_ne_strategy, nash_dqn_first_player_ne_strategy)
                    self.kl_list[1].append(kl_dist)

                elif id_state < 9:
                    ne_strategy = nash_strategies[id_state]
                    oracle_first_player_ne_strategy = ne_strategy[0]
                    nash_dqn_first_player_ne_strategy = dists[0][0]
                    kl_dist = kl(oracle_first_player_ne_strategy, nash_dqn_first_player_ne_strategy)
                    self.kl_list[2].append(kl_dist)

                print(f'KL: {kl_dist}， id_state: {id_state}')
                with open('./data/nash_kl3.npy', 'wb') as f:
                    np.save(f, self.kl_list)

        else:
            actions = np.random.randint(self.action_dims, size=(state.shape[0], self.num_agents))
        
        if self.num_envs == 1:
            actions = actions[0]  # list of actions to its item
        else:
            actions = np.array(actions).T  # to shape: (agents, envs, action_dim)
        return actions

    def compute_nash(self, q_values, return_dist=False):
        """
        Return actions as Nash equilibrium of given payoff matrix, shape: [env, agent]
        """
        q_table = q_values.reshape(-1, self.action_dims,  self.action_dims)
        all_actions = []
        all_dists = []
        for qs in q_table:  # iterate over envs
            try:
                # ne = NashEquilibriaSolver(qs)
                # ne = ne[0]  # take the first Nash equilibria found
                # print(np.linalg.det(qs))
                # ne = NashEquilibriumSolver(qs)
                # ne = NashEquilibriumLPSolver(qs)
                # ne = NashEquilibriumCVXPYSolver(qs)
                # ne = NashEquilibriumGUROBISolver(qs)
                ne = NashEquilibriumECOSSolver(qs)

            except:  # some cases NE cannot be solved
                print('No Nash solution for: ', np.linalg.det(qs), qs)
                ne = self.num_agents*[1./qs.shape[0]*np.ones(qs.shape[0])]  # use uniform distribution if no NE is found
            
            actions = []
                
            all_dists.append(ne)
            for dist in ne:  # iterate over agents
                try:
                    sample_hist = np.random.multinomial(1, dist)  # return one-hot vectors as sample from multinomial
                except:
                    print('Not a valid distribution from Nash equilibrium solution.')
                    print(sum(ne[0]), sum(ne[1]))
                    print(qs, ne)
                    print(dist)
                a = np.where(sample_hist>0)
                actions.append(a)
            all_actions.append(np.array(actions).reshape(-1))
        if return_dist:
            return all_dists
        else:
            return np.array(all_actions), all_dists

    def compute_cce(self, q_values, return_dist=False):
        """
        Return actions as coarse correlated equilibrium of given payoff matrix, shape: [env, agent]
        """
        q_table = q_values.reshape(-1, self.action_dims,  self.action_dims)
        all_actions = []
        all_dists = []
        for qs in q_table:  # iterate over envs
            try:
                _, _, jnt_probs = CoarseCorrelatedEquilibriumLPSolver(qs)

            except:  # some cases NE cannot be solved
                print('No CCE solution for: ', np.linalg.det(qs), qs)
                jnt_probs = 1./(qs.shape[0]*qs.shape[1])*np.ones(qs.shape[0]*qs.shape[1])  # use uniform distribution if no NE is found
            
            try:
                sample_hist = np.random.multinomial(1, jnt_probs)  # a joint probability matrix for all players
            except:
                print('Not a valid distribution from Nash equilibrium solution.')
                print(sum(jnt_probs), sum(abs(jnt_probs)))
                print(qs, jnt_probs)
            sample_hist = sample_hist.reshape(self.action_dims,  self.action_dims)
            a = np.where(sample_hist>0)  # the actions for two players
            all_actions.append(np.array(a).reshape(-1))
            all_dists.append(jnt_probs)
        if return_dist:
            return all_dists
        else:
            return np.array(all_actions)

    def update(self):
        state, action, reward, next_state, done = self.buffer.sample(self.batch_size)
        weights = torch.ones(self.batch_size)

        state = torch.FloatTensor(np.float32(state)).to(self.device)
        next_state = torch.FloatTensor(np.float32(next_state)).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(np.float32(done)).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        # Q-Learning with target network
        q_values = self.model(state)
        # target_next_q_values_ = self.model(next_state)  # or use this one
        target_next_q_values_ = self.target(next_state)
        target_next_q_values = target_next_q_values_.detach().cpu().numpy()

        action_dim = int(np.sqrt(q_values.shape[-1])) # for two-symmetric-agent case only
        action = torch.LongTensor([a[0]*action_dim+a[1] for a in action]).to(self.device)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        # compute CCE or NE
        # if args.cce: # Coarse Correlated Equilibrium
        #     cce_dists = self.compute_cce(target_next_q_values, return_dist=True)
        #     target_next_q_values_ = target_next_q_values_.reshape(-1, action_dim, action_dim)
        #     cce_dists_  = torch.FloatTensor(cce_dists).to(self.device)
        #     next_q_value = torch.einsum('bij,bij->b', cce_dists_, target_next_q_values_)

        # else: # Nash Equilibrium
        nash_dists = self.compute_nash(target_next_q_values, return_dist=True)  # get the mixed strategy Nash rather than specific actions
        target_next_q_values_ = target_next_q_values_.reshape(-1, action_dim, action_dim)
        nash_dists_  = torch.FloatTensor(nash_dists).to(self.device)
        next_q_value = torch.einsum('bk,bk->b', torch.einsum('bj,bjk->bk', nash_dists_[:, 0], target_next_q_values_), nash_dists_[:, 1])
        # print(next_q_value, target_next_q_values_)

        expected_q_value = reward + (self.gamma ** self.multi_step) * next_q_value * (1 - done)
        
        # Huber Loss
        loss = F.smooth_l1_loss(q_value, expected_q_value.detach(), reduction='none')
        loss = (loss * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.update_cnt % self.target_update_interval == 0:
            self.update_target(self.model, self.target)
            self.update_cnt = 0
        self.update_cnt += 1

        return loss.item()

class NashDQNBase(DQNBase):
    """
    Nash-DQN for parallel env sampling

    parameters
    ---------
    env         environment(openai gym)
    """
    def __init__(self, env, net_args, number_envs=2):
        super().__init__(env, net_args)
        self.number_envs = number_envs
        try:
            self._observation_shape = tuple(map(operator.add, env.observation_space.shape, env.observation_space.shape)) # double the shape
            self._action_shape = (env.action_space.n)**2
        except:
            self._observation_shape = tuple(map(operator.add, env.observation_space[0].shape, env.observation_space[0].shape)) # double the shape
            self._action_shape = (env.action_space[0].n)**2
        self._construct_net(env, net_args)

    def _construct_net(self, env, net_args):
            input_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape = self._observation_shape)
            output_space = gym.spaces.Discrete(self._action_shape)
            if len(self._observation_shape) <= 1: # not image
                self.net = get_model('mlp')(input_space, output_space, net_args, model_for='discrete_q')
            else:
                self.net = get_model('cnn')(input_space, output_space, net_args, model_for='discrete_q')
