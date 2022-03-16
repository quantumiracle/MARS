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

class NashDQNFactorized(DQN):
    """
    Nash-DQN algorithm
    """
    def __init__(self, env, args):
        super().__init__(env, args)
        self.num_envs = args.num_envs
        self.q_net_1 = NashDQNFactorizedBase(env, args.net_architecture, args.num_envs, two_side_obs = args.marl_spec['global_state']).to(self.device)
        self.q_net_2 = NashDQNFactorizedBase(env, args.net_architecture, args.num_envs, two_side_obs = args.marl_spec['global_state']).to(self.device)
        self.target_q_net_1 = copy.deepcopy(self.q_net_1).to(self.device)
        self.target_q_net_2 = copy.deepcopy(self.q_net_2).to(self.device)

        if args.num_process > 1:
            self.q_net_1.share_memory()
            self.q_net_2.share_memory()
            self.target_q_net_1.share_memory()
            self.target_q_net_2.share_memory()

        self.num_agents = env.num_agents[0] if isinstance(env.num_agents, list) else env.num_agents
        try:
            self.action_dims = env.action_space[0].n
        except:
            self.action_dims = env.action_space.n
        self.env = env
        # don't forget to instantiate an optimizer although there is one in DQN
        self.nash_optimizer = choose_optimizer(args.optimizer)(list(self.q_net_1.parameters())+list(self.q_net_2.parameters()), lr=float(args.learning_rate))
        self.dqn_optimizer = choose_optimizer(args.optimizer)(list(self.q_net_1.parameters())+list(self.q_net_2.parameters()), lr=float(args.learning_rate))

        if DEBUG:
            self.debugger = Debugger(env, "./data/factorized_nash_dqn_test/nash_dqn_simple_mdp_log.pkl")

    def model(self, state):
        """
        Merged Q table: Q(s,a,b)
        """
        q1 = self.q_net_1(state)  # shape: (#batch, #action1)
        q2 = self.q_net_2(state)  # shape: (#batch, #action2)
        merged_q = 0.5*(q1[:, :, None] - q2[:, None])  # shape: (#batch, #action1, #action2)

        return merged_q


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
            try: # nash computation may report error and terminate the process
                actions, dists, ne_vs = self.compute_nash(q_values)
            except:
                print("Invalid nash computation.")
                actions = np.random.randint(self.action_dims, size=(state.shape[0], self.num_agents))

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
                    else:
                        test_states = torch.FloatTensor(np.repeat(np.arange(total_states_num), 2, axis=0).reshape(-1, 2)).to(self.device)
                    ne_q_vs = self.model(test_states) # Nash Q values
                    ne_q_vs = ne_q_vs.view(self.env.env.max_transition, self.env.env.num_states, self.action_dims, self.action_dims).detach().cpu().numpy()

                    self.debugger.compare_with_oracle(state, dists, ne_vs, ne_q_vs, verbose=False)

        else:
            actions = np.random.randint(self.action_dims, size=(state.shape[0], self.num_agents))  # (envs, agents)
        
        if self.num_envs == 1:
            actions = actions[0]  # list of actions to its item
        else:
            actions = np.array(actions).T  # to shape: (agents, envs, action_dim)
        return actions

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

    def update(self):
        state, action, reward, next_state, done = self.buffer.sample(self.batch_size)

        state = torch.FloatTensor(np.float32(state)).to(self.device)
        next_state = torch.FloatTensor(np.float32(next_state)).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(np.float32(done)).to(self.device)

        # invididual DQN loss
        def DQN_loss(model, target, state, action, reward, next_state, done, multi_step):
            q = model(state)
            q = q.gather(1, action.unsqueeze(1)).squeeze(1)
            next_q = target(next_state)
            next_q = torch.max(next_q, dim=-1)
            target_q = reward + (self.gamma ** multi_step) * next_q * (1 - done)
            loss = F.mse_loss(q, target_q.detach(), reduction='none')
            return loss.mean()
        a1 = torch.LongTensor(action[:, 0]).to(self.device)
        a2 = torch.LongTensor(action[:, 1]).to(self.device)
        q1_loss = DQN_loss(self.q_net_1, self.target_q_net_1, state, a1, reward, next_state, done, self.multi_step)
        q2_loss = DQN_loss(self.q_net_2, self.target_q_net_2, state, a2, -reward, next_state, done, self.multi_step)

        self.dqn_optimizer.zero_grad()
        q1_loss.backward()
        q2_loss.backward()
        self.dqn_optimizer.step()

        # Nash loss
        q_values = self.model(state)
        target_next_q_values_ = self.target(next_state)
        target_next_q_values = target_next_q_values_.detach().cpu().numpy()
        action_ = torch.LongTensor([a[0]*self.action_dims+a[1] for a in action]).to(self.device)  # encode the actions from both players to be one scalar
        q_value = q_values.gather(1, action_.unsqueeze(1)).squeeze(1)
        try: # nash computation may encounter error and terminate the process
            _, next_q_value = self.compute_nash(target_next_q_values, update=True)
        except: 
            print("Invalid nash computation.")
            next_q_value = np.zeros_like(reward)
        next_q_value  = torch.FloatTensor(next_q_value).to(self.device)
        expected_q_value = reward + (self.gamma ** self.multi_step) * next_q_value * (1 - done)
        nash_loss = F.mse_loss(q_value, expected_q_value.detach(), reduction='none')
        nash_loss = nash_loss.mean()

        self.nash_optimizer.zero_grad()
        nash_loss.backward()
        self.nash_optimizer.step()

        if self.update_cnt % self.target_update_interval == 0:
            self.update_target(self.model, self.target)
        self.update_cnt += 1
        return nash_loss.item()

class NashDQNFactorizedBase(DQNBase):
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
        except:
            if two_side_obs:
                self._observation_shape = tuple(map(operator.add, env.observation_space[0].shape, env.observation_space[0].shape)) # double the shape
            else:
                self._observation_shape = env.observation_space[0].shape
        self._action_shape = env.action_space[0].n
        self._construct_net(env, net_args)

    def _construct_net(self, env, net_args):
            input_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape = self._observation_shape)
            output_space = gym.spaces.Discrete(self._action_shape)
            if len(self._observation_shape) <= 1: # not 3d image
                self.net = get_model('mlp')(input_space, output_space, net_args, model_for='discrete_q')
            else:
                self.net = get_model('cnn')(input_space, output_space, net_args, model_for='discrete_q')
