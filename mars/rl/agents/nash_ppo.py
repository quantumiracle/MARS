import copy
from math import log
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from .agent import Agent
from ..common.networks import MLP, CNN, get_model
from ..common.rl_utils import choose_optimizer
from mars.utils.typing import List, Tuple, StateType, ActionType, SampleType, SingleEnvMultiAgentSampleType
import operator
import gym

class NashPPO(Agent):
    """ Nash-PPO agorithm for environments with discrete action space.
    """
    def __init__(self, env, args):
        super().__init__(env, args)
        self.learning_rate = args.learning_rate
        self.gamma = float(args.algorithm_spec['gamma'])
        self.lmbda = float(args.algorithm_spec['lambda'])
        self.eps_clip = float(args.algorithm_spec['eps_clip'])
        self.K_epoch = args.algorithm_spec['K_epoch']
        self.GAE = args.algorithm_spec['GAE']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if isinstance(env.observation_space, list):  # when using parallel envs
            observation_space = env.observation_space[0]
        else:
            observation_space = env.observation_space
        self.policies, self.values, self.feature_nets = [], [], []
        try:
            merged_action_space_dim = env.action_space.n + env.action_space.n
        except:
            merged_action_space_dim = env.action_space[0].n + env.action_space[0].n
        merged_action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(merged_action_space_dim,))
        # args.net_architecture['policy']
        args_param = args.net_architecture
        args_param['policy']['output_activation'] = 'Linear'
        args_param['hidden_dim_list'] = [64, 64]
        # import pdb; pdb.set_trace()
        #softmax = nn.functional.softmax()
        if len(observation_space.shape) <= 1:
            for _ in range(2):
                tmp_action = gym.spaces.Discrete(env.action_space.n * 2)
                self.feature_nets.append(MLP(env.observation_space, [tmp_action, tmp_action], args_param, model_for='discrete_policy').to(self.device))
                # import pdb; pdb.set_trace()
                self.policies.append(nn.Sequential(nn.Linear(env.action_space.n * 2, env.action_space.n),nn.Softmax()).to(self.device))
                self.values.append(nn.Linear(env.action_space.n * 2, 1).to(self.device))
            # self.common_layers = MLP(env.observation_space, merged_action_space, args.net_architecture['value'], model_for='discrete_q').to(self.device)
            self.common_layers = nn.Linear(env.action_space.n * 4, 1).to(self.device)
        else:
            for _ in range(2):
                self.policies.append(CNN(env.observation_space, env.action_space, args.net_architecture['policy'],
                                         model_for='discrete_policy').to(self.device))
                self.values.append(
                    CNN(env.observation_space, env.action_space, args.net_architecture['value'], model_for='value').to(
                        self.device))
            # self.common_layers = CNN(env.observation_space, merged_action_space, args.net_architecture['value'], model_for='discrete_q').to(self.device)
            self.common_layers = CNN(env.observation_space, env.action_space, args.net_architecture['value'],
                                     model_for='value').to(self.device)
        if args.num_process > 1:
            self.policies = [policy.share_memory() for policy in self.policies]
            self.values = [value.share_memory() for value in self.values]
            self.common_layers.share_memory()
        policy_params, value_params, common_val_params, feature_net_param = [], [], [], []
        for p, v, z in zip(self.feature_nets, self.policies, self.values):
            policy_params += list(p.parameters())
            value_params += list(v.parameters())
            common_val_params += list(z.parameters())
        feature_net_param += list(self.common_layers.parameters())
        # TODO a single optimizer for two nets may be problematic
        self.optimizer = choose_optimizer(args.optimizer)(policy_params + value_params + common_val_params + feature_net_param, lr=float(args.learning_rate))
        self.common_layer_optimizer = choose_optimizer(args.optimizer)(policy_params + value_params + common_val_params + feature_net_param,
                                                                        lr=float(args.learning_rate))
        self.mseLoss = nn.MSELoss()
        self._num_channel = args.num_envs * (env.num_agents if isinstance(env.num_agents, int) else env.num_agents[
            0])  # env.num_agents is a list when using parallel envs
        self.data = [[] for _ in range(self._num_channel)]

    def pi(
            self,
            x: List[StateType],
            idx
    ) -> List[ActionType]:
        """ Forward the policy network.
        :param x: input of the policy network, i.e. the state
        :type x: List[StateType]
        :return: the actions
        :rtype: List[ActionType]
        """
        return self.policies[idx].forward(x)

    def v(
            self,
            x: List[StateType],
            idx
    ) -> List[float]:
        """ Forward the value network.
        :param x: input of the value network, i.e. the state
        :type x: List[StateType]
        :return: a list of values for each state
        :rtype: List[float]
        """
        return self.values[idx].forward(x)

    def reinit(self, ):
        self.policy.reinit()
        self.policy_old.reinit()
        self.value.reinit()

    def store(self, transitions: SampleType) -> None:
        """ Store samples in batch.
        :param transitions: a list of samples from different environments (if using parallel env)
        :type transitions: SampleType
        """
        # self.data.append(transition)
        # self.data.extend(transitions)
        # If several transitions are pushed at the same time,
        # they are not from the same trajectory, therefore they need
        # to be stored separately since PPO is on-policy.
        for i, transition in enumerate(transitions):  # iterate over the list
            self.data[i].append(transition)

    def choose_action(
            self,
            s: StateType,
            Greedy: bool = False
    ) -> List[ActionType]:
        """Choose action give state.
        :param s: observed state from the agent
        :type s: List[StateType]
        :param Greedy: whether adopt greedy policy (no randomness for exploration) or not, defaults to False
        :type Greedy: bool, optional
        :return: the actions
        :rtype: List[ActionType]
        """
        actions = []
        actions = []
        logprobs = []
        if Greedy:
            for policy, feature_net, state_per_agent in zip(self.policies, self.feature_nets, s):
                feature = feature_net(torch.from_numpy(np.array(state_per_agent)).unsqueeze(0).float().to(self.device))
                prob = policy(feature).squeeze()  # make sure input state shape is correct
                dist = Categorical(prob)
                a = dist.sample()
                logprob = dist.log_prob(a)
                actions.append(a.detach().cpu().numpy())
            actions = np.array(actions)
            return actions
        else:
            for policy, feature_net, state_per_agent in zip(self.policies, self.feature_nets, s):
                feature = feature_net(torch.from_numpy(np.array(state_per_agent)).unsqueeze(0).float().to(self.device))
                prob = policy(feature).squeeze()  # make sure input state shape is correct
                dist = Categorical(prob)
                a = dist.sample()
                logprob = dist.log_prob(a)
                actions.append(a.detach().cpu().numpy())
                logprobs.append(logprob.detach().cpu().numpy())
            actions = np.array(actions)
            logprobs = np.array(logprobs)
            return actions, logprobs

    def make_batch(
            self,
            data: SampleType
    ) -> SingleEnvMultiAgentSampleType:
        """ Reshape the data and put it into the computational device, cpu or gpu.
        :param data: unstructured data
        :type data: SampleType
        :return: structured data
        :rtype: SingleEnvMultiAgentSampleType
        """
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for trajectory in data:
            s, a, r, s_prime, prob_a, done = trajectory
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            prob_a_lst.append(prob_a)
            done_mask = 0 if done else 1
            done_lst.append(done_mask)
        a_lst = np.array(a_lst)
        r_lst = np.array(r_lst)
        s_lst = np.array(s_lst)
        s_prime_lst  = np.array(s_prime_lst)
        done_mask = np.array(done_mask)
        prob_a_lst = np.array(prob_a_lst)
        # found this step take some time for Pong (not ram), even if no parallel no multiagent
        s, a, r, s_prime, prob_a, done_mask = torch.tensor(s_lst, dtype=torch.float).to(self.device), torch.tensor(
            a_lst).to(self.device), \
                                              torch.tensor(r_lst, dtype=torch.float).to(self.device), torch.tensor(
            s_prime_lst, dtype=torch.float).to(self.device), \
                                              torch.tensor(prob_a_lst, dtype=torch.float).to(self.device), torch.tensor(
            done_lst, dtype=torch.float).to(self.device)
        return s, a, r, s_prime, prob_a, done_mask

    def update(self):
        total_loss = 0.
        self.data = [x for x in self.data if x]  # remove empty
        for data in self.data:  # iterate over data from different environments
            s, a, r, s_prime, oldlogprob, done_mask = self.make_batch(data)
            # done_mask = done_mask.repeat(6,1).T
            # need to prcess the samples, separate for agents
            s_ = s.view(s.shape[0], 2, -1)
            s_prime_ = s_prime.view(s_prime.shape[0], 2, -1)
            for iteration_ in range(self.K_epoch):
                loss = 0.0
                ppo_loss_total = 0.0
                feature_x_list = []
                feature_x_prime_list = []
                # standard PPO
                for i in range(2):  # for each agent
                    feature_x = self.feature_nets[i](s_[:, i, :])
                    feature_x_prime = self.feature_nets[i](s_prime_[:, i, :])
                    # feature_x_list.append(feature_x)
                    # feature_x_prime_list.append(feature_x_prime)
                    vs = self.v(feature_x, i)  # take the state for the specific agent
                    # use generalized advantage estimation (GAE)
                    vs_prime = self.v(feature_x_prime, i).squeeze(dim=-1)
                    # import pdb; pdb.set_trace()
                    #assert vs_prime.shape == done_mask.shape
                    r = r.detach()
                    vs_target = r[:, i] + self.gamma * vs_prime * done_mask
                    assert vs_prime.shape == done_mask.shape
                    delta = vs_target - vs.squeeze(dim=-1)
                    # delta = delta.detach()
                    advantage_lst = []
                    advantage = 0.0
                    for delta_t in torch.flip(delta,
                                              [-1]):  # reverse the delta along the time sequence in an episodic data
                        advantage = self.gamma * self.lmbda * advantage + delta_t
                        advantage_lst.append(advantage)
                    advantage_lst.reverse()
                    advantage = torch.tensor(advantage_lst, dtype=torch.float).to(self.device)
                    advantage = (advantage - advantage.mean()) / (
                                advantage.std() + 1e-5)  # this can have significant improvement (efficiency, stability) on performance
                    advantage = advantage.detach()
                    # value and policy loss for one agent
                    pi = self.pi(feature_x, i)
                    dist = Categorical(pi)
                    dist_entropy = dist.entropy()
                    logprob = dist.log_prob(a[:, i])
                    ratio = torch.exp(logprob - oldlogprob[:, i])
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
                    ppo_loss = -torch.min(surr1, surr2) + F.mse_loss(vs.squeeze(dim=-1),
                                                                     vs_target.detach()) - 0.01 * dist_entropy  # TODO vec + scalar + vec, is this valid?
                    ppo_loss = ppo_loss.mean()
                    ppo_loss_total += ppo_loss
                    self.optimizer.zero_grad()
                    ppo_loss.backward()
                    self.optimizer.step()
                    # loss += ppo_loss
                    total_loss += ppo_loss.item()
                # loss for common layers (value function)
                # import pdb; pdb.set_trace()
                feature_x_list = []
                feature_x_prime_list = []
                # standard PPO
                for i in range(2):  # for each agent
                    feature_x = self.feature_nets[i](s_[:, i, :])
                    feature_x_prime = self.feature_nets[i](s_prime_[:, i, :])
                    feature_x_list.append(feature_x)
                    feature_x_prime_list.append(feature_x_prime)
                vs = self.common_layers(torch.cat(feature_x_list, axis=1))  # TODO just use the first state (assume it has full info)
                vs_prime = self.common_layers(torch.cat(feature_x_prime_list, axis=1)).squeeze(
                    dim=-1)  # TODO just use the first state (assume it has full info)
                assert vs_prime.shape == done_mask.shape
                vs_target = r[:, 0] + self.gamma * vs_prime * done_mask  # r is the first player's here
                common_layer_loss = F.mse_loss(vs.squeeze(dim=-1), vs_target.detach()).mean()
                # calculate generalized advantage with common layer value
                delta = vs_target - vs.squeeze(dim=-1)
                delta = delta.detach()
                advantage_lst = []
                advantage = 0.0
                for delta_t in torch.flip(delta, [-1]):  # reverse the delta along the time sequence in an episodic data
                    advantage = self.gamma * self.lmbda * advantage + delta_t
                    advantage_lst.append(advantage)
                advantage_lst.reverse()
                advantage = torch.tensor(advantage_lst, dtype=torch.float).to(self.device)
                advantage = (advantage - advantage.mean()) / (
                            advantage.std() + 1e-5)  # this can have significant improvement (efficiency, stability) on performance
                advantage = advantage.detach()
                # TODO can the following be written once? (error in previous trial)
                ratio_list = []
                for i in range(2):  # get the ratio for both
                    pi = self.pi(feature_x_list[i], i)
                    dist = Categorical(pi)
                    dist_entropy = dist.entropy()
                    logprob = dist.log_prob(a[:, i])
                    ratio = torch.exp(logprob - oldlogprob[:, i])
                    ratio_list.append(ratio)  # the ratios need to be newly computed to have policy gradients
                surr1 = ratio_list[0] * ratio_list[1].detach() * advantage
                surr2 = torch.clamp(ratio_list[0] * (ratio_list[1].detach()), 1 - self.eps_clip, 1 + self.eps_clip)
                policy_loss1 = -torch.min(surr1, surr2).mean()
                # self.optimizer.zero_grad()
                # policy_loss1.backward()
                # self.optimizer.step()
                ratio_list = []
                for i in range(2):  # get the ratio for both
                    pi = self.pi(feature_x_list[i], i)
                    dist = Categorical(pi)
                    dist_entropy = dist.entropy()
                    logprob = dist.log_prob(a[:, i])
                    ratio = torch.exp(logprob - oldlogprob[:, i])
                    ratio_list.append(ratio)  # the ratios need to be newly computed to have policy gradients
                surr1 = ratio_list[0].detach() * ratio_list[1] * advantage
                surr2 = torch.clamp((ratio_list[0].detach()) * ratio_list[1], 1 - self.eps_clip, 1 + self.eps_clip)
                policy_loss2 = torch.min(surr1, surr2).mean()
                # self.optimizer.zero_grad()
                # policy_loss2.backward()
                # common_layer_loss.backward()
                # self.optimizer.step()
                # self.common_layer_optimizer.zero_grad()
                # common_layer_loss.backward()
                # self.common_layer_optimizer.step()
                # if (iteration_ % 500 == 0):
                #     print('@@')
                #     print('indpendent loss = {}, p1 = {}, p2 = {}, common = {}'.format(ppo_loss_total,policy_loss1 * 0.08, policy_loss2 * 0.08, common_layer_loss * 1.0))
                loss = 0.08 * (policy_loss1 + policy_loss2) + 1.0 * (common_layer_loss)
                #loss += ppo_loss_total
                # self.common_layer_optimizer.zero_grad()
                # loss.backward()
                # self.common_layer_optimizer.step()
                # import pdb; pdb.set_trace()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
        self.data = [[] for _ in range(self._num_channel)]
        # Copy new weights into old policy:
        return total_loss

    def save_model(self, path=None):
        try:  # for PyTorch >= 1.7 to be compatible with loading models from any lower version
            for i, (pi, v, feature) in enumerate(zip(self.policies, self.values, self.feature_nets)):
                torch.save(pi.state_dict(), path + f'_{i}_policy', _use_new_zipfile_serialization=False)
                torch.save(v.state_dict(), path + f'_{i}_value', _use_new_zipfile_serialization=False)
                torch.save(feature.state_dict(), path + f'_{i}_feature', _use_new_zipfile_serialization=False)
            torch.save(self.common_layers.state_dict(), path + f'_common', _use_new_zipfile_serialization=False)
        except:
            for i, (pi, v, feature) in enumerate(zip(self.policies, self.values, self.feature_nets)):
                torch.save(pi.state_dict(), path + f'_{i}_policy')
                torch.save(v.state_dict(), path + f'_{i}_value')
                torch.save(feature.state_dict(), path + f'_{i}_feature')
            torch.save(self.common_layers.state_dict(), path + f'_common')
            
    def load_model(self, path=None, eval=True):
        for i, (pi, v, feature) in enumerate(zip(self.policies, self.values, self.feature_nets)):
            pi.load_state_dict(torch.load(path + f'_{i}_policy'))
            v.load_state_dict(torch.load(path + f'_{i}_value'))
            feature.load_state_dict(torch.load(path + f'_{i}_feature'))
        self.common_layers.load_state_dict(torch.load(path + f'_common'))