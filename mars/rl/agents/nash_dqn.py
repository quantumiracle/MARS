import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import gym
import operator
import random, copy
import pickle
from ..common.rl_utils import choose_optimizer, EpsilonScheduler
from ..common.networks import NetBase, get_model
from .dqn import DQN, DQNBase
from mars.equilibrium_solver import NashEquilibriumECOSSolver, NashEquilibriumMWUSolver, NashEquilibriumParallelMWUSolver
import time

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
        self.cnt = 0
        self.save_interval = 10
        self.logging = {'num_states_per_step': self.num_states_per_step,
                        'max_transition': self.max_transition,
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
        print(self.oracle_nash_q_values)
    def compare_with_oracle(self, state, dists, ne_vs, verbose=False):
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
                br_v = np.min(nash_dqn_first_player_ne_strategy@ne_q)  # best response value (value against best response), reflects exploitability of learned Nash 
                kl_dist = kl(oracle_first_player_ne_strategy, nash_dqn_first_player_ne_strategy)
                self.kl_dist_list[j].append(kl_dist)
                mse_v = float((ne_v - ne_vs)**2) # squared error of Nash values (predicted and oracle)
                print(ne_v, ne_vs)
                self.mse_v_list[j].append(mse_v)
                mse_exp = float((ne_v - br_v)**2)  # the target value of best response value (exploitability) should be the Nash value
                self.mse_exp_list[j].append(mse_exp)


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

    def dump_log(self,):
        with open(self.log_path, "wb") as f:
            pickle.dump(self.logging, f)    

class NashDQN(DQN):
    """
    Nash-DQN algorithm
    """
    def __init__(self, env, args):
        super().__init__(env, args)
        self.num_envs = args.num_envs
        self.model = NashDQNBase(env, args.net_architecture, args.num_envs, two_side_obs = args.marl_spec['global_state']).to(self.device)
        self.target = copy.deepcopy(self.model).to(self.device)

        if args.multiprocess:
            self.model.share_memory()
            self.target.share_memory()
        self.num_agents = env.num_agents[0] if isinstance(env.num_agents, list) else env.num_agents
        try:
            self.action_dims = env.action_space[0].n
        except:
            self.action_dims = env.action_space.n
        # don't forget to instantiate an optimizer although there is one in DQN
        self.optimizer = choose_optimizer(args.optimizer)(self.model.parameters(), lr=float(args.learning_rate))
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.95)    
        # self.schedulers.append(lr_scheduler)

        if DEBUG:
            self.debugger = Debugger(env, "./data/nash_dqn_test/nash_dqn_simple_mdp_log.pkl")

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
            try: # nash computation may report error and terminate the process
                actions, dists, ne_vs = self.compute_nash(q_values)
            except:
                print("Invalid nash computation.")
                actions = np.random.randint(self.action_dims, size=(state.shape[0], self.num_agents))

            if DEBUG: ## test on arbitrary MDP
                self.debugger.compare_with_oracle(state, dists, ne_vs, verbose=True)

        else:
            actions = np.random.randint(self.action_dims, size=(state.shape[0], self.num_agents))  # (envs, agents)
        
        if self.num_envs == 1:
            actions = actions[0]  # list of actions to its item
        else:
            actions = np.array(actions).T  # to shape: (agents, envs, action_dim)
        return actions

    def compute_nash_deprecated(self, q_values, update=False):
        """
        Return actions as Nash equilibrium of given payoff matrix, shape: [env, agent]
        """
        q_tables = q_values.reshape(-1, self.action_dims,  self.action_dims)
        all_actions = []
        all_dists = []
        all_ne_values = []
        for qs in q_tables:  # iterate over envs
            # Solve Nash equilibrium with solver
            try:
                # ne = NashEquilibriaSolver(qs)
                # ne = ne[0]  # take the first Nash equilibria found
                # print(np.linalg.det(qs))
                # ne = NashEquilibriumSolver(qs)
                # ne = NashEquilibriumLPSolver(qs)
                # ne = NashEquilibriumCVXPYSolver(qs)
                # ne = NashEquilibriumGUROBISolver(qs)
                ne, ne_v = NashEquilibriumECOSSolver(qs)
                ne_v = ne[0]@qs@ne[1].T
                # ne, ne_v = NashEquilibriumMWUSolver(qs)
            except:  # some cases NE cannot be solved
                print('No Nash solution for: ', np.linalg.det(qs), qs)
                ne = self.num_agents*[1./qs.shape[0]*np.ones(qs.shape[0])]  # use uniform distribution if no NE is found
                ne_v = 0
                
            all_dists.append(ne)
            all_ne_values.append(ne_v)

            # Sample actions from Nash strategies
            actions = []
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

        if update:
            return all_dists, all_ne_values
        else: # return samples actions, nash strategies, nash values
            return np.array(all_actions), all_dists, all_ne_values

    def compute_nash(self, q_values, update=False):
        q_tables = q_values.reshape(-1, self.action_dims,  self.action_dims)
        all_actions = []
        all_dists = []
        all_ne_values = []
        all_dists, all_ne_values = NashEquilibriumParallelMWUSolver(q_tables)

        if update:
            return all_dists, all_ne_values
        else:
            # Sample actions from Nash strategies
            for ne in all_dists:
                actions = []
                for dist in ne:  # iterate over agents
                    try:
                        sample_hist = np.random.multinomial(1, dist)  # return one-hot vectors as sample from multinomial
                    except:
                        print('Not a valid distribution from Nash equilibrium solution.')
                        print(sum(ne[0]), sum(ne[1]))
                        print(dist)
                    a = np.where(sample_hist>0)
                    actions.append(a)
                all_actions.append(np.array(actions).reshape(-1))

        return np.array(all_actions), all_dists, all_ne_values


    def compute_cce(self, q_values, return_dist=False):
        """
        Return actions as coarse correlated equilibrium of given payoff matrix, shape: [env, agent]
        """
        q_tables = q_values.reshape(-1, self.action_dims,  self.action_dims)
        all_actions = []
        all_dists = []
        for qs in q_tables:  # iterate over envs
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

        state = torch.FloatTensor(np.float32(state)).to(self.device)
        next_state = torch.FloatTensor(np.float32(next_state)).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(np.float32(done)).to(self.device)

        # Q-Learning with target network
        q_values = self.model(state)
        target_next_q_values_ = self.target(next_state)
        target_next_q_values = target_next_q_values_.detach().cpu().numpy()

        action_dim = int(np.sqrt(q_values.shape[-1])) # for two-symmetric-agent case only
        action_ = torch.LongTensor([a[0]*action_dim+a[1] for a in action]).to(self.device)
        q_value = q_values.gather(1, action_.unsqueeze(1)).squeeze(1)

        # compute CCE or NE
        # if args.cce: # Coarse Correlated Equilibrium
        #     cce_dists = self.compute_cce(target_next_q_values, return_dist=True)
        #     target_next_q_values_ = target_next_q_values_.reshape(-1, action_dim, action_dim)
        #     cce_dists_  = torch.FloatTensor(cce_dists).to(self.device)
        #     next_q_value = torch.einsum('bij,bij->b', cce_dists_, target_next_q_values_)

        # else: # Nash Equilibrium
        # try: # nash computation may report error and terminate the process
        #     nash_dists, _ = self.compute_nash(target_next_q_values, update=True)  # get the mixed strategy Nash rather than specific actions
        # except: # take a uniform distribution instead
        #     print("Invalid nash computation.")
        #     nash_dists = np.ones((*action.shape, self.action_dims))/float(self.action_dims)
        # target_next_q_values_ = target_next_q_values_.reshape(-1, action_dim, action_dim)
        # nash_dists_  = torch.FloatTensor(nash_dists).to(self.device)
        # next_q_value = torch.einsum('bk,bk->b', torch.einsum('bj,bjk->bk', nash_dists_[:, 0], target_next_q_values_), nash_dists_[:, 1])

        try: # nash computation may encounter error and terminate the process
            _, next_q_value = self.compute_nash(target_next_q_values, update=True)
        except: 
            print("Invalid nash computation.")
            next_q_value = np.zeros_like(reward)
        next_q_value  = torch.FloatTensor(next_q_value).to(self.device)
        
        expected_q_value = reward + (self.gamma ** self.multi_step) * next_q_value * (1 - done)

        # Huber Loss
        # loss = F.smooth_l1_loss(q_value, expected_q_value.detach(), reduction='none')
        loss = F.mse_loss(q_value, expected_q_value.detach(), reduction='none')

        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.update_cnt % self.target_update_interval == 0:
            self.update_target(self.model, self.target)
            # self.update_cnt = 0
        self.update_cnt += 1
        # print(self.update_cnt)
        return loss.item()

class NashDQNBase(DQNBase):
    """
    Nash-DQN for parallel env sampling

    parameters
    ---------
    env         environment(openai gym)
    """
    def __init__(self, env, net_args, number_envs=2, two_side_obs=True):
        super().__init__(env, net_args)
        self.number_envs = number_envs
        try:
            if two_side_obs:
                self._observation_shape = tuple(map(operator.add, env.observation_space.shape, env.observation_space.shape)) # double the shape
            else:
                self._observation_shape = env.observation_space.shape
            self._action_shape = (env.action_space.n)**2
        except:
            if two_side_obs:
                self._observation_shape = tuple(map(operator.add, env.observation_space[0].shape, env.observation_space[0].shape)) # double the shape
            else:
                self._observation_shape = env.observation_space[0].shape
            self._action_shape = (env.action_space[0].n)**2
        self._construct_net(env, net_args)

    def _construct_net(self, env, net_args):
            input_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape = self._observation_shape)
            output_space = gym.spaces.Discrete(self._action_shape)
            if len(self._observation_shape) <= 1: # not 3d image
                self.net = get_model('mlp')(input_space, output_space, net_args, model_for='discrete_q')
            else:
                self.net = get_model('cnn')(input_space, output_space, net_args, model_for='discrete_q')
