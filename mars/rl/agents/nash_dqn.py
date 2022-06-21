import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import gym
import operator
import random, copy
from ..common.rl_utils import choose_optimizer, EpsilonScheduler
from ..common.networks import NetBase, get_model
from .dqn import DQN, DQNBase
from .debug import Debugger, to_one_hot
from mars.equilibrium_solver import NashEquilibriumECOSSolver, NashEquilibriumMWUSolver, NashEquilibriumParallelMWUSolver

DEBUG = False
class NashDQN(DQN):
    """
    Nash-DQN algorithm
    """
    def __init__(self, env, args):
        super().__init__(env, args)
        self.num_envs = args.num_envs

        if args.num_process > 1:
            self.model.share_memory()
            self.target.share_memory()
        self.num_agents = env.num_agents[0] if isinstance(env.num_agents, list) else env.num_agents
        self.env = env
        self.args = args

        # don't forget to instantiate an optimizer although there is one in DQN
        self.optimizer = choose_optimizer(args.optimizer)(self.model.parameters(), lr=float(args.learning_rate))
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.95)    
        # self.schedulers.append(lr_scheduler)

        if DEBUG:
            self.debugger = Debugger(env, "./data/nash_dqn_test/nash_dqn_simple_mdp_log_target_itr100_5step_1033.pkl")

    def _init_model(self, env, args):
        """Overwrite DQN's models

        :param env: environment
        :type env: object
        :param args: arguments
        :type args: dict
        """
        self.model = NashDQNBase(env, args.net_architecture, args.num_envs, two_side_obs = args.marl_spec['global_state']).to(self.device)
        print(self.model)
        self.target = copy.deepcopy(self.model).to(self.device)

    def choose_action(self, state, Greedy=False, epsilon=None):
        if Greedy:
            epsilon = 0.
        elif epsilon is None:
            epsilon = self.epsilon_scheduler.get_epsilon()
        if not isinstance(state, torch.Tensor):
            state = torch.Tensor(state).to(self.device)
        if self.args.ram:
            if self.num_envs == 1: # state: (agents, state_dim)
                state = state.unsqueeze(0).view(1, -1) # change state from (agents, state_dim) to (1, agents*state_dim)
            else: # state: (agents, envs, state_dim)
                state = torch.transpose(state, 0, 1) # to state: (envs, agents, state_dim)
                state = state.view(state.shape[0], -1) # to state: (envs, agents*state_dim)
        else:  # image-based input
            if self.num_envs == 1: # state: (agents, C, H, W)
                state = state.unsqueeze(0).view(1, -1, state.shape[-2], state.shape[-1])  #   (1, agents*C, H, W)

            else: # state: (agents, envs, C, H, W)
                state = torch.transpose(state, 0, 1) # state: (envs, agents, C, H, W)
                state = state.view(state.shape[0], -1, state.shape[-2], state.shape[-1]) # state: (envs, agents*C, H, W)

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
                actions = np.random.randint(self.action_dim, size=(state.shape[0], self.num_agents))

            if DEBUG: ## test on arbitrary MDP
                if self.update_cnt % 111 == 0: # skip some steps, 111 is not divided by number of transitions
                    total_states_num = self.env.env.num_states*self.env.env.max_transition
                    if self.env.env.OneHotObs:
                        range = self.env.env.num_states*(self.env.env.max_transition+1)
                        states = np.arange(total_states_num)
                        one_hot_states = []
                        for s in states:
                            one_hot_states.append(to_one_hot(s, range))
                        test_states = torch.FloatTensor(np.repeat(one_hot_states, 2, axis=0).reshape(-1, 2*range)).to(self.device)
                        # print(test_states)
                    else:
                        test_states = torch.FloatTensor(np.repeat(np.arange(total_states_num), 2, axis=0).reshape(-1, 2)).to(self.device)
                    ne_q_vs = self.model(test_states) # Nash Q values
                    ne_q_vs = ne_q_vs.view(self.env.env.max_transition, self.env.env.num_states, self.action_dim, self.action_dim).detach().cpu().numpy()

                    self.debugger.compare_with_oracle(state, dists, ne_vs, ne_q_vs, verbose=False)

        else:
            actions = np.random.randint(self.action_dim, size=(state.shape[0], self.num_agents))  # (envs, agents)
        
        if self.num_envs == 1:
            actions = actions[0]  # list of actions to its item
        else:
            actions = np.array(actions).T  # to shape: (agents, envs, action_dim)
        return actions

    def compute_nash_deprecated(self, q_values, update=False):
        """
        Return actions as Nash equilibrium of given payoff matrix, shape: [env, agent]
        """
        q_tables = q_values.reshape(-1, self.action_dim,  self.action_dim)
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
        q_tables = q_values.reshape(-1, self.action_dim,  self.action_dim)
        all_actions = []
        all_dists = []
        all_ne_values = []
        # import time
        # time.sleep(0.01)

        # all_dists, all_ne_values = NashEquilibriumParallelMWUSolver(q_tables)
        for q_table in q_tables:
            dist, value = NashEquilibriumECOSSolver(q_table)
            all_dists.append(dist)
            all_ne_values.append(value)

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
        q_tables = q_values.reshape(-1, self.action_dim,  self.action_dim)
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
            sample_hist = sample_hist.reshape(self.action_dim,  self.action_dim)
            a = np.where(sample_hist>0)  # the actions for two players
            all_actions.append(np.array(a).reshape(-1))
            all_dists.append(jnt_probs)
        if return_dist:
            return all_dists
        else:
            return np.array(all_actions)

    def update(self):
        DoubleTrick = False
        state, action, reward, next_state, done = self.buffer.sample(self.batch_size)

        state = torch.FloatTensor(np.float32(state)).to(self.device)
        next_state = torch.FloatTensor(np.float32(next_state)).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(np.float32(done)).to(self.device)

        # Q-Learning with target network
        q_values = self.model(state)
        target_next_q_values_ = self.model(next_state) if DoubleTrick else self.target(next_state)
        target_next_q_values = target_next_q_values_.detach().cpu().numpy()

        action_ = torch.LongTensor([a[0]*self.action_dim+a[1] for a in action]).to(self.device)
        q_value = q_values.gather(1, action_.unsqueeze(1)).squeeze(1)

        # compute CCE or NE
        # if args.cce: # Coarse Correlated Equilibrium
        #     cce_dists = self.compute_cce(target_next_q_values, return_dist=True)
        #     target_next_q_values_ = target_next_q_values_.reshape(-1, action_dim, action_dim)
        #     cce_dists_  = torch.FloatTensor(cce_dists).to(self.device)
        #     next_q_value = torch.einsum('bij,bij->b', cce_dists_, target_next_q_values_)

        # else: # Nash Equilibrium
        try: # nash computation may encounter error and terminate the process
            next_dist, next_q_value = self.compute_nash(target_next_q_values, update=True)
        except: 
            print("Invalid nash computation.")
            next_q_value = np.zeros_like(reward)

        if DoubleTrick: # calculate next_q_value using double DQN trick
            next_dist = np.array(next_dist)  # shape: (#batch, #agent, #action)
            target_next_q_values = target_next_q_values.reshape((-1, self.action_dim, self.action_dim))
            left_multi = np.einsum('na,nab->nb', next_dist[:, 0], target_next_q_values) # shape: (#batch, #action)
            next_q_value = np.einsum('nb,nb->n', left_multi, next_dist[:, 1]) 

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
        self.update_cnt += 1
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
