from heapq import merge
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
from .nash_dqn import NashDQNBase

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
        self.nash_q = NashDQNBase(env, args.net_architecture, args.num_envs, two_side_obs = args.marl_spec['global_state']).to(self.device)
        self.target_q_net_1 = copy.deepcopy(self.q_net_1).to(self.device)
        self.target_q_net_2 = copy.deepcopy(self.q_net_2).to(self.device)
        self.target_nash_q = copy.deepcopy(self.nash_q).to(self.device)

        if args.num_process > 1:
            self.q_net_1.share_memory()
            self.q_net_2.share_memory()
            self.nash_q.share_memory()
            self.target_q_net_1.share_memory()
            self.target_q_net_2.share_memory()
            self.target_nash_q.share_memory()

        self.num_agents = env.num_agents[0] if isinstance(env.num_agents, list) else env.num_agents
        try:
            self.action_dims = env.action_space[0].n
        except:
            self.action_dims = env.action_space.n
        self.env = env
        # don't forget to instantiate an optimizer although there is one in DQN
        self.nash_optimizer = choose_optimizer(args.optimizer)(list(self.q_net_1.parameters())+list(self.q_net_2.parameters())+list(self.nash_q.parameters()), lr=0.2*float(args.learning_rate)) # smaller lr for fine-tuning with nash loss
        self.dqn_optimizer = choose_optimizer(args.optimizer)(list(self.q_net_1.parameters())+list(self.q_net_2.parameters()), lr=float(args.learning_rate))

        if DEBUG:
            self.debugger = Debugger(env, "./data/factorized_nash_dqn_test/nash_dqn_simple_mdp_log.pkl")
        
    def _get_nash_q(self, state, single_side_q1, single_side_q2, nash_q_correction):
        """
        Merged Q table: Q(s,a,b) = 0.5*(Q(s,a)-Q(s,b)) + delta_Q(s,a,b)
        """
        # sum of single side Q
        q1 = single_side_q1(state)  # shape: (#batch, #action1)
        q2 = single_side_q2(state)  # shape: (#batch, #action2)
        merged_q = 0.5*(q1[:, :, None] - q2[:, None])  # 0.5*(Q(s,a)-Q(s,b)); shape: (#batch, #action1, #action2)
        merged_q = merged_q.view(merged_q.shape[0], -1)  # reshape: (#batch, #action1 * #action2)

        delta_nash_q = nash_q_correction(state) # shape: (#batch, #action1 * #action2)

        nash_q = merged_q + delta_nash_q # Q(s,a,b) = 0.5*(Q(s,a)-Q(s,b)) + delta_Q(s,a,b)

        return nash_q     

    def model_(self, state):
        """
        Merged Q table: Q(s,a,b) = 0.5*(Q(s,a)-Q(s,b)) + delta_Q(s,a,b)
        """
        nash_q = self._get_nash_q(state, self.q_net_1, self.q_net_2, self.nash_q)
        return nash_q

    def target_(self, state):
        """
        Merged Q table: Q(s,a,b)
        """
        target_nash_q = self._get_nash_q(state, self.target_q_net_1, self.target_q_net_2, self.target_nash_q)
        return target_nash_q

    ### FOR TEST PURPOSE ONLY ###
    ### two side DQN ###
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
            q1 = self.q_net_1(state)  # shape: (#batch, #action1)
            q2 = self.q_net_2(state)  # shape: (#batch, #action2)   
            a1 = torch.argmax(q1, dim=-1, keepdim=True) # shape: (#batch, 1)
            a2 = torch.argmax(q2, dim=-1, keepdim=True) # shape: (#batch, 1)
            actions = torch.hstack((a1, a2)).detach().cpu().numpy()
    
        else:
            actions = np.random.randint(self.action_dims, size=(state.shape[0], self.num_agents))  # (envs, agents)
        
        if self.num_envs == 1:
            actions = actions[0]  # list of actions to its item
        else:
            actions = np.array(actions).T  # to shape: (agents, envs, (action_dim=1))

        return actions


    # def choose_action(self, state, Greedy=False, epsilon=None):
    #     if Greedy:
    #         epsilon = 0.
    #     elif epsilon is None:
    #         epsilon = self.epsilon_scheduler.get_epsilon()
    #     if not isinstance(state, torch.Tensor):
    #         state = torch.Tensor(state).to(self.device)
    #     if self.num_envs == 1: # state: (agents, state_dim)
    #         state = state.unsqueeze(0).view(1, -1) # change state from (agents, state_dim) to (1, agents*state_dim)
    #     else: # state: (agents, envs, state_dim)
    #         state = torch.transpose(state, 0, 1) # to state: (envs, agents, state_dim)
    #         state = state.view(state.shape[0], -1) # to state: (envs, agents*state_dim)

    #     if random.random() > epsilon:  # NoisyNet does not use e-greedy
    #         with torch.no_grad():
    #             q_values = self.model_(state).detach().cpu().numpy()  # needs state: (batch, agents*state_dim)
    #         try: # nash computation may report error and terminate the process
    #             actions, dists, ne_vs = self.compute_nash(q_values)
    #         except:
    #             print("Error: Invalid nash computation in choose_action function.")
    #             actions = np.random.randint(self.action_dims, size=(state.shape[0], self.num_agents))
    #         if np.isnan(actions).any():
    #             print("Error: Nan action value in Nash computation is derived in the choose_action function.")
    #             actions = np.random.randint(self.action_dims, size=(state.shape[0], self.num_agents))

    #         if DEBUG: ## test on arbitrary MDP
    #             if self.update_cnt % 111 == 0: # skip some steps, 111 is not divided by number of transitions
    #                 total_states_num = self.env.env.num_states*self.env.env.max_transition
    #                 if self.env.env.OneHotObs:
    #                     range = self.env.env.num_states*(self.env.env.max_transition+1)
    #                     states = np.arange(total_states_num)
    #                     one_hot_states = []
    #                     for s in states:
    #                         one_hot_states.append(to_one_hot(s, range))
    #                     test_states = torch.FloatTensor(np.repeat(one_hot_states, 2, axis=0).reshape(-1, 2*range)).to(self.device)
    #                 else:
    #                     test_states = torch.FloatTensor(np.repeat(np.arange(total_states_num), 2, axis=0).reshape(-1, 2)).to(self.device)
    #                 ne_q_vs = self.model_(test_states) # Nash Q values
    #                 ne_q_vs = ne_q_vs.view(self.env.env.max_transition, self.env.env.num_states, self.action_dims, self.action_dims).detach().cpu().numpy()

    #                 self.debugger.compare_with_oracle(state, dists, ne_vs, ne_q_vs, verbose=False)

    #     else:
    #         actions = np.random.randint(self.action_dims, size=(state.shape[0], self.num_agents))  # (envs, agents)
        
    #     if self.num_envs == 1:
    #         actions = actions[0]  # list of actions to its item
    #     else:
    #         actions = np.array(actions).T  # to shape: (agents, envs, action_dim)

    #     return actions

    def compute_nash(self, q_values, update=False):
        q_tables = q_values.reshape(-1, self.action_dims,  self.action_dims)
        all_actions = []
        all_dists = []
        all_ne_values = []
        all_dists, all_ne_values = NashEquilibriumParallelMWUSolver(q_tables)

        if update:
            return all_dists, all_ne_values #  Nash distributions, Nash values
        else:
            # Sample actions from Nash strategies
            for ne in all_dists:
                actions = []
                for dist in ne:  # iterate over agents
                    try:
                        sample_hist = np.random.multinomial(1, dist)  # return one-hot vectors as sample from multinomial
                    except:
                        print('Error: Not a valid distribution from Nash equilibrium solution.')
                        print(sum(ne[0]), sum(ne[1]))
                        print(dist)
                    a = np.where(sample_hist>0)
                    actions.append(a)
                all_actions.append(np.array(actions).reshape(-1))

            return np.array(all_actions), all_dists, all_ne_values  # Nash actions, Nash distributions, Nash values

    def update(self):
        state, action, reward, next_state, done = self.buffer.sample(self.batch_size)

        state = torch.FloatTensor(np.float32(state)).to(self.device)
        next_state = torch.FloatTensor(np.float32(next_state)).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(np.float32(done)).to(self.device)

        # invididual DQN loss calculation
        def DQN_loss(model, target, state, action, reward, next_state, done, multi_step):
            q = model(state)
            q = q.gather(1, action.unsqueeze(1)).squeeze(1)
            next_q = target(next_state)
            next_q = torch.max(next_q, dim=-1)[0]
            target_q = reward + (self.gamma ** multi_step) * next_q * (1 - done)
            loss = F.mse_loss(q, target_q.detach(), reduction='none')
            return loss.mean()

        a1 = action[:, 0]
        a2 = action[:, 1]
        q1_loss = DQN_loss(self.q_net_1, self.target_q_net_1, state, a1, reward, next_state, done, self.multi_step)
        q2_loss = DQN_loss(self.q_net_2, self.target_q_net_2, state, a2, -reward, next_state, done, self.multi_step)

        self.dqn_optimizer.zero_grad()
        q1_loss.backward()
        q2_loss.backward()
        self.dqn_optimizer.step()

        # Nash loss
        q_values = self.model_(state)
        target_next_q_values_ = self.target_(next_state)
        target_next_q_values = target_next_q_values_.detach().cpu().numpy()
        action_ = torch.LongTensor([a[0]*self.action_dims+a[1] for a in action]).to(self.device)  # encode the actions from both players to be one scalar
        q_value = q_values.gather(1, action_.unsqueeze(1)).squeeze(1)
        try: # nash computation may encounter error and terminate the process
            _, next_q_value = self.compute_nash(target_next_q_values, update=True)
            next_q_value  = torch.FloatTensor(next_q_value).to(self.device)

        except: 
            print("Error: Invalid nash computation in the update function.")
            next_q_value = torch.zeros_like(reward)

        if torch.isnan(next_q_value).any():
            print("Error: Nan Nash value in Nash computation is derived in the udpate function.")
            next_q_value = torch.zeros_like(reward)

        expected_q_value = reward + (self.gamma ** self.multi_step) * next_q_value * (1 - done)
        nash_loss = F.mse_loss(q_value, expected_q_value.detach(), reduction='none')
        nash_loss = nash_loss.mean()

        # self.nash_optimizer.zero_grad()
        # nash_loss.backward()
        # self.nash_optimizer.step()

        if self.update_cnt % self.target_update_interval == 0:
            self.update_target(self.q_net_1, self.target_q_net_1)
            self.update_target(self.q_net_2, self.target_q_net_2)
            self.update_target(self.nash_q, self.target_nash_q)
        self.update_cnt += 1
        return nash_loss.item()

    def save_model(self, path):
        try:  # for PyTorch >= 1.7 to be compatible with loading models from any lower version
            torch.save(self.q_net_1.state_dict(), path+'_qnet1', _use_new_zipfile_serialization=False) 
            torch.save(self.q_net_2.state_dict(), path+'_qnet2', _use_new_zipfile_serialization=False)
            torch.save(self.nash_q.state_dict(), path+'_nashq', _use_new_zipfile_serialization=False)
            torch.save(self.target_q_net_1.state_dict(), path+'_target_qnet1', _use_new_zipfile_serialization=False) 
            torch.save(self.target_q_net_2.state_dict(), path+'_target_qnet2', _use_new_zipfile_serialization=False)
            torch.save(self.target_nash_q.state_dict(), path+'_target_nashq', _use_new_zipfile_serialization=False)
        except:  # for lower versions
            torch.save(self.q_net_1.state_dict(), path+'_qnet1') 
            torch.save(self.q_net_2.state_dict(), path+'_qnet2')
            torch.save(self.nash_q.state_dict(), path+'_nashq')
            torch.save(self.target_q_net_1.state_dict(), path+'_target_qnet1') 
            torch.save(self.target_q_net_2.state_dict(), path+'_target_qnet2')
            torch.save(self.target_nash_q.state_dict(), path+'_target_nashq')

    def load_model(self, path, eval=True):
        self.q_net_1.load_state_dict(torch.load(path+'_qnet1'))
        self.q_net_2.load_state_dict(torch.load(path+'_qnet2'))
        self.nash_q.load_state_dict(torch.load(path+'_nashq'))
        self.target_q_net_1.load_state_dict(torch.load(path+'_target_qnet1'))
        self.target_q_net_2.load_state_dict(torch.load(path+'_target_qnet2'))
        self.target_nash_q.load_state_dict(torch.load(path+'_target_nashq'))

        if eval:
            self.q_net_1.eval()
            self.q_net_2.eval()
            self.nash_q.eval()
            self.target_q_net_1.eval()
            self.target_q_net_2.eval()
            self.target_nash_q.eval()

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
                self._action_shape = env.action_space.n
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
            if len(self._observation_shape) <= 1: # not 2d image
                self.net = get_model('mlp')(input_space, output_space, net_args, model_for='discrete_q')
            else:
                self.net = get_model('cnn')(input_space, output_space, net_args, model_for='discrete_q')
