import copy
from math import log
import gym
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal, MultivariateNormal
import numpy as np
from .agent import Agent
from ..common.networks import MLP, CNN, get_model
from ..common.rl_utils import choose_optimizer
from mars.utils.typing import List, Tuple, StateType, ActionType, SampleType, SingleEnvMultiAgentSampleType


def NashPPO(env, args):
    """ The function returns a proper class for Nash PPO algorithm,
    according to the action space of the environment.

    :param env: environment instance
    :type env: object
    :param args: arguments
    :type args: ConfigurationDict
    """    
    if isinstance(env.action_space, list):
        if isinstance(env.action_space[0], gym.spaces.Box):
            return NashPPOContinuous(env, args)
        else:
            return NashPPODiscrete(env, args)
    else:
        if isinstance(env.action_space, gym.spaces.Box):
            return NashPPOContinuous(env, args)
        else:
            return NashPPODiscrete(env, args)
            
class NashPPOBase(Agent):
    """ Nash-PPO agorithm base agent.
    """
    def __init__(self, env, args):
        super().__init__(env, args)
        self.learning_rate = args.learning_rate
        self.gamma = float(args.algorithm_spec['gamma'])
        self.lmbda = float(args.algorithm_spec['lambda'])
        self.eps_clip = float(args.algorithm_spec['eps_clip'])
        self.K_epoch = args.algorithm_spec['K_epoch']
        self.GAE = args.algorithm_spec['GAE']
        self.policy_loss_coeff = args.algorithm_spec['policy_loss_coeff']
        self.max_grad_norm = float(args.algorithm_spec['max_grad_norm'])
        self.entropy_coeff = float(args.algorithm_spec['entropy_coeff'])
        self.vf_coeff = float(args.algorithm_spec['vf_coeff'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._init_model(env, args)
        self.args = args
        print(f'Feature networks: ', self.feature_nets,
                'Policy networks: ', self.policies,
                'Value networks: ', self.values,
                'Common layers: ', self.common_layers)

        policy_params, value_params, common_val_params, feature_net_param = [], [], [], []
        for p, v, z in zip(self.feature_nets, self.policies, self.values):
            policy_params += list(p.parameters())
            value_params += list(v.parameters())
            common_val_params += list(z.parameters())

        feature_net_param += list(self.common_layers.parameters())
        self.all_params = policy_params + value_params + common_val_params + feature_net_param
        self.optimizer = choose_optimizer(args.optimizer)(self.all_params, lr=float(args.learning_rate))
        self.mseLoss = nn.MSELoss()
        self._num_channel = args.num_envs * (env.num_agents if isinstance(env.num_agents, int) else env.num_agents[0])  # env.num_agents is a list when using parallel envs
        self.data = [[] for _ in range(self._num_channel)]

    def _init_model(self, env, args):
        self.policies, self.values, self.feature_nets = [], [], []
        # try:
        #     merged_action_space_dim = env.action_space.n + env.action_space.n
        #     [low, high] = [env.action_space.low, env.action_space.high]
        # except:
        #     merged_action_space_dim = env.action_space[0].n + env.action_space[0].n
        #     [low, high] = [env.action_space[0].low, env.action_space[0].high]
        # merged_action_space = gym.spaces.Box(low=low, high=high, shape=(merged_action_space_dim,))
        # merged_action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(merged_action_space_dim,))

        if len(self.observation_space.shape) <= 1:
            feature_space = self.observation_space
            double_feature_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape = (feature_space.shape[0]*2,)) # TODO other types of spaces like discrete etc
            for _ in range(env.num_agents):
                self.feature_nets.append(MLP(env.observation_space, feature_space, args.net_architecture['feature'], model_for='feature').to(self.device))
                self.policies.append(MLP(feature_space, env.action_space, args.net_architecture['policy'], model_for=self.policy_type).to(self.device))
                self.values.append(MLP(feature_space, env.action_space, args.net_architecture['value'], model_for='value').to(self.device))
            
            self.common_layers = MLP(double_feature_space, env.action_space, args.net_architecture['value'], model_for='value').to(self.device)

        else:
            feature_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape = (256,))
            double_feature_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape = (feature_space.shape[0]*2,)) # TODO other types of spaces like discrete etc

            for _ in range(env.num_agents):
                self.feature_nets.append(CNN(env.observation_space, feature_space, args.net_architecture['feature'], model_for='feature').to(self.device))
                self.policies.append(MLP(feature_space, env.action_space, args.net_architecture['policy'], model_for=self.policy_type).to(self.device))
                self.values.append(MLP(feature_space, env.action_space, args.net_architecture['value'], model_for='value').to(self.device))

            self.common_layers = MLP(double_feature_space, env.action_space, args.net_architecture['value'], model_for='value').to(self.device)
        
        if args.num_process > 1:
            self.policies = [policy.share_memory() for policy in self.policies]
            self.values = [value.share_memory() for value in self.values]
            self.common_layers.share_memory()


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
        # TODO
        self.policy.reinit()
        self.value.reinit()

    def store(self, transitions: SampleType, max_capacity = 1e4) -> None:
        """ Store samples in batch.
        :param transitions: a list of samples from different environments (if using parallel env)
        :type transitions: SampleType
        """
        # If several transitions are pushed at the same time,
        # they are not from the same trajectory, therefore they need
        # to be stored separately since PPO is on-policy.
        for i, transition in enumerate(transitions):  # iterate over the list
            self.data[i].append(transition)
            if len(self.data[1]) > max_capacity:  # clear the buffer
                self.data[i] = []

    def choose_action(
            self,
            s: StateType,
            Greedy: bool = False
    ) -> List[ActionType]:
        pass

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

    def update(self,):
        pass

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


class NashPPODiscrete(NashPPOBase):
    """ Nash-PPO agorithm for environments with discrete action space.
    """
    def __init__(self, env, args):
        super().__init__(env, args)

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
                if self.args.ram or self.args.num_envs == 1:
                    feature = feature_net(torch.from_numpy(np.array(state_per_agent)).unsqueeze(0).float().to(self.device))
                else:
                    feature = feature_net(torch.from_numpy(np.array(state_per_agent)).float().to(self.device))
                prob = policy(feature).squeeze()  # make sure input state shape is correct
                a = torch.argmax(prob, dim=-1)
                # dist = Categorical(prob)
                # a = dist.sample()
                # logprob = dist.log_prob(a)
                actions.append(a.detach().cpu().numpy())
            actions = np.array(actions)
            return actions
        else:
            for policy, feature_net, state_per_agent in zip(self.policies, self.feature_nets, s):
                if self.args.ram:
                    feature = feature_net(torch.from_numpy(np.array(state_per_agent)).unsqueeze(0).float().to(self.device))
                else:
                    if len(state_per_agent.shape) <= 3:
                        feature = feature_net(torch.from_numpy(np.array(state_per_agent)).unsqueeze(0).float().to(self.device))
                    else:
                        feature = feature_net(torch.from_numpy(np.array(state_per_agent)).float().to(self.device))
                prob = policy(feature).squeeze()  # make sure input state shape is correct
                dist = Categorical(prob)
                a = dist.sample()
                logprob = dist.log_prob(a)
                actions.append(a.detach().cpu().numpy())
                logprobs.append(logprob.detach().cpu().numpy())
            actions = np.array(actions)
            logprobs = np.array(logprobs)
            return actions, logprobs

    def update(self):
        total_loss = 0.
        infos = {}
        self.data = [x for x in self.data if x]  # remove empty
        for data in self.data:  # iterate over data from different environments
            s, a, r, s_prime, oldlogprob, done_mask = self.make_batch(data)

            # need to prcess the samples, separate for agents
            if self.args.ram:
                s_ = s.view(s.shape[0], 2, -1)
                s_prime_ = s_prime.view(s_prime.shape[0], 2, -1)
            else:
                s_ = s  # shape: (batch, agents, envs, C, H, W)
                s_prime_ = s_prime

            for _ in range(self.K_epoch):
                loss = 0.0
                ppo_loss_total = 0.0
                feature_x_list = []
                feature_x_prime_list = []

                # standard PPO
                for i in range(2):  # for each agent
                    with torch.no_grad():
                        # shared feature extraction
                        if self.args.ram:
                            feature_x = self.feature_nets[i](s_[:, i, :])
                            feature_x_prime = self.feature_nets[i](s_prime_[:, i, :])
                        else:
                            feature_x = self.feature_nets[i](s_[:, i])
                            feature_x_prime = self.feature_nets[i](s_prime_[:, i])

                        vs = self.v(feature_x, i)  # take the state for the specific agent
                        vs_prime = self.v(feature_x_prime, i).squeeze(dim=-1)
                        assert vs_prime.shape == done_mask.shape
                        r = r.detach()
                        vs_target = r[:, i] + self.gamma * vs_prime * done_mask
                        delta = vs_target - vs.squeeze(dim=-1)
                        advantage_lst = []
                        advantage = 0.0
                        for delta_t in torch.flip(delta, [-1]):  # reverse the delta along the time sequence in an episodic data
                            advantage = self.gamma * self.lmbda * advantage + delta_t
                            advantage_lst.append(advantage)
                        advantage_lst.reverse()
                        advantage = torch.tensor(advantage_lst, dtype=torch.float).to(self.device)
                        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)  # this can have significant improvement (efficiency, stability) on performance
                        advantage = advantage.detach()

                    # value and policy loss for one agent
                    pi = self.pi(feature_x, i)
                    dist = Categorical(pi)
                    dist_entropy = dist.entropy()
                    logprob = dist.log_prob(a[:, i])
                    ratio = torch.exp(logprob - oldlogprob[:, i])
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
                    policy_loss = -torch.min(surr1, surr2)
                    v_loss = F.mse_loss(vs.squeeze(dim=-1), vs_target.detach())
                    ppo_loss = policy_loss + self.vf_coeff * v_loss - self.entropy_coeff * dist_entropy  # TODO vec + scalar + vec, is this valid?
                    ppo_loss = ppo_loss.mean()
                    ppo_loss_total += ppo_loss
                    self.optimizer.zero_grad()
                    ppo_loss.backward()
                    nn.utils.clip_grad_norm_(self.all_params, self.max_grad_norm)
                    self.optimizer.step()
                    total_loss += ppo_loss.item()

                # loss for common layers (value function)
                feature_x_list = []
                feature_x_prime_list = []
                # standard PPO
                for i in range(2):  # for each agent
                    feature_x = self.feature_nets[i](s_[:, i, :])
                    feature_x_prime = self.feature_nets[i](s_prime_[:, i, :])
                    feature_x_list.append(feature_x)
                    feature_x_prime_list.append(feature_x_prime)
                vs = self.common_layers(torch.cat(feature_x_list, axis=1))  # TODO just use the first state (assume it has full info)
                vs_prime = self.common_layers(torch.cat(feature_x_prime_list, axis=1)).squeeze(dim=-1)  # TODO just use the first state (assume it has full info)
                assert vs_prime.shape == done_mask.shape
                vs_target = r[:, 0] + self.gamma * vs_prime * done_mask  # r is the first player's here
                common_layer_loss = F.mse_loss(vs.squeeze(dim=-1), vs_target.detach()).mean()

                # calculate generalized advantage with common layer value
                with torch.no_grad():
                    delta = vs_target - vs.squeeze(dim=-1)
                    delta = delta.detach()
                    advantage_lst = []
                    advantage = 0.0
                    for delta_t in torch.flip(delta, [-1]):  # reverse the delta along the time sequence in an episodic data
                        advantage = self.gamma * self.lmbda * advantage + delta_t
                        advantage_lst.append(advantage)
                    advantage_lst.reverse()
                    advantage = torch.tensor(advantage_lst, dtype=torch.float).to(self.device)
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)  # this can have significant improvement (efficiency, stability) on performance
                    advantage = advantage.detach()

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

                loss = self.policy_loss_coeff * (policy_loss1 + policy_loss2) + 1.0 * (common_layer_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.all_params, self.max_grad_norm)
                self.optimizer.step()
                total_loss += loss.item()

        infos[f'PPO policy loss player {i}'] = policy_loss
        infos[f'PPO value loss player {i}'] = v_loss
        infos[f'PPO total loss player {i}'] = ppo_loss
        infos[f'policy entropy player {i}'] = dist_entropy
        infos[f'Nash value loss'] = common_layer_loss
        infos[f'Nash policy loss player 0'] = policy_loss1
        infos[f'Nash policy loss player 1'] = policy_loss2

        self.data = [[] for _ in range(self._num_channel)]

        return total_loss, infos


class NashPPOContinuous(NashPPOBase):
    """ Nash-PPO agorithm for environments with discrete action space.
    """
    def __init__(self, env, args):
        super().__init__(env, args)

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
                feature = feature_net(torch.from_numpy(np.array(state_per_agent)).unsqueeze(0).float().to(self.device))  # make sure input state shape is correct
                logits = policy(feature)

                if len(logits.shape) > 2:
                    logits = logits.squeeze()
                mean = logits[:, :self.action_dim]
                var = logits[:, self.action_dim:].exp()
                
                # feature = feature_net(torch.from_numpy(np.array(state_per_agent)).unsqueeze(0).float().to(self.device))
                # prob = policy(feature).squeeze()  # make sure input state shape is correct
                # a = torch.argmax(prob, dim=-1)
                # a = mean
                actions.append(mean.detach().cpu().numpy())
            actions = np.array(actions)
            return actions
        else:
            for policy, feature_net, state_per_agent in zip(self.policies, self.feature_nets, s):
                feature = feature_net(torch.from_numpy(np.array(state_per_agent)).unsqueeze(0).float().to(self.device))
                logits = policy(feature)
                if len(logits.shape) > 2:
                    logits = logits.squeeze()
                mean = logits[:, :self.action_dim]
                var = logits[:, self.action_dim:].exp()
                cov = torch.diag_embed(var)
                dist = MultivariateNormal(mean, cov)
                # dist = Normal(mean, var**0.5)
                a = dist.sample()
                logprob = dist.log_prob(a)       

                actions.append(a.detach().cpu().numpy())
                logprobs.append(logprob.detach().cpu().numpy())
            actions = np.array(actions)
            logprobs = np.array(logprobs)
            return actions, logprobs

    def update(self):
        infos = {}
        total_loss = 0.
        self.data = [x for x in self.data if x]  # remove empty
        for data in self.data:  # iterate over data from different environments
            s, a, r, s_prime, oldlogprob, done_mask = self.make_batch(data)

            # need to prcess the samples, separate for agents
            if self.args.ram:
                s_ = s.view(s.shape[0], 2, -1)
                s_prime_ = s_prime.view(s_prime.shape[0], 2, -1)
            else:
                s_ = s  # shape: (batch, agents, envs, C, H, W)
                s_prime_ = s_prime   
            a = a.view(a.shape[0], 2, -1)

            for _ in range(self.K_epoch):
                loss = 0.0
                ppo_loss_total = 0.0
                feature_x_list = []
                feature_x_prime_list = []

                # standard PPO
                for i in range(2):  # for each agent
                    # shared feature extraction
                    if self.args.ram:
                        feature_x = self.feature_nets[i](s_[:, i, :])
                        feature_x_prime = self.feature_nets[i](s_prime_[:, i, :])
                    else:
                        feature_x = self.feature_nets[i](s_[:, i])
                        feature_x_prime = self.feature_nets[i](s_prime_[:, i])                        

                    vs = self.v(feature_x, i)  # take the state for the specific agent
                    vs_prime = self.v(feature_x_prime, i).squeeze(dim=-1)
                    assert vs_prime.shape == done_mask.shape
                    r = r.detach()
                    vs_target = r[:, i] + self.gamma * vs_prime * done_mask
                    delta = vs_target - vs.squeeze(dim=-1)
                    advantage_lst = []
                    advantage = 0.0
                    for delta_t in torch.flip(delta, [-1]):  # reverse the delta along the time sequence in an episodic data
                        advantage = self.gamma * self.lmbda * advantage + delta_t
                        advantage_lst.append(advantage)
                    advantage_lst.reverse()
                    advantage = torch.tensor(advantage_lst, dtype=torch.float).to(self.device)
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)  # this can have significant improvement (efficiency, stability) on performance
                    advantage = advantage.detach()

                    # value and policy loss for one agent
                    # pi = self.pi(feature_x, i)
                    # dist = Categorical(pi)

                    logits = self.pi(feature_x, i)
                    if len(logits.shape) > 2:
                        logits = logits.squeeze()
                    mean = logits[:, :self.action_dim]
                    var = logits[:, self.action_dim:].exp()
                    cov = torch.diag_embed(var)
                    dist = MultivariateNormal(mean, cov)

                    dist_entropy = dist.entropy()
                    logprob = dist.log_prob(a[:, i].squeeze())  # for multivariate normal, sum of log_prob is produce of prob
                    ratio = torch.exp(logprob - oldlogprob[:, i])
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
                    # print(surr1.shape, surr2.shape, vs.squeeze(dim=-1).shape, vs_target.shape, ratio.shape, advantage.shape, dist_entropy.shape)
                    policy_loss = -torch.min(surr1, surr2)
                    v_loss = F.mse_loss(vs.squeeze(dim=-1), vs_target.detach())
                    ppo_loss = policy_loss + self.vf_coeff * v_loss - self.entropy_coeff * dist_entropy  # TODO vec + scalar + vec, is this valid?
                    ppo_loss = ppo_loss.mean()
                    ppo_loss_total += ppo_loss
                    self.optimizer.zero_grad()
                    ppo_loss.backward()
                    nn.utils.clip_grad_norm_(self.all_params, self.max_grad_norm)
                    self.optimizer.step()
                    total_loss += ppo_loss.item()
                    infos[f'PPO policy loss player {i}'] = policy_loss
                    infos[f'PPO value loss player {i}'] = v_loss
                    infos[f'PPO total loss player {i}'] = ppo_loss
                    infos[f'policy entropy player {i}'] = dist_entropy

                # loss for common layers (value function)
                feature_x_list = []
                feature_x_prime_list = []
                # standard PPO
                for i in range(2):  # for each agent
                    feature_x = self.feature_nets[i](s_[:, i, :])
                    feature_x_prime = self.feature_nets[i](s_prime_[:, i, :])
                    feature_x_list.append(feature_x)
                    feature_x_prime_list.append(feature_x_prime)
                vs = self.common_layers(torch.cat(feature_x_list, axis=1))  # TODO just use the first state (assume it has full info)
                vs_prime = self.common_layers(torch.cat(feature_x_prime_list, axis=1)).squeeze(dim=-1)  # TODO just use the first state (assume it has full info)
                assert vs_prime.shape == done_mask.shape
                vs_target = r[:, 0] + self.gamma * vs_prime * done_mask  # r is the first player's here
                common_layer_loss = F.mse_loss(vs.squeeze(dim=-1), vs_target.detach()).mean()
                infos[f'Nash value loss'] = common_layer_loss

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
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)  # this can have significant improvement (efficiency, stability) on performance
                advantage = advantage.detach()

                ratio_list = []
                for i in range(2):  # get the ratio for both
                    # pi = self.pi(feature_x_list[i], i)
                    # dist = Categorical(pi)

                    logits = self.pi(feature_x_list[i], i)
                    if len(logits.shape) > 2:
                        logits = logits.squeeze()
                    mean = logits[:, :self.action_dim]
                    var = logits[:, self.action_dim:].exp()
                    cov = torch.diag_embed(var)
                    dist = MultivariateNormal(mean, cov)
                    dist_entropy = dist.entropy()
                    logprob = dist.log_prob(a[:, i])
                    ratio = torch.exp(logprob - oldlogprob[:, i])
                    ratio_list.append(ratio)  # the ratios need to be newly computed to have policy gradients
                surr1 = ratio_list[0] * ratio_list[1].detach() * advantage
                surr2 = torch.clamp(ratio_list[0] * (ratio_list[1].detach()), 1 - self.eps_clip, 1 + self.eps_clip)
                policy_loss1 = -torch.min(surr1, surr2).mean()
                infos[f'Nash policy loss player 1'] = policy_loss1

                ratio_list = []
                for i in range(2):  # get the ratio for both
                    # pi = self.pi(feature_x_list[i], i)
                    # dist = Categorical(pi)

                    logits = self.pi(feature_x_list[i], i)
                    if len(logits.shape) > 2:
                        logits = logits.squeeze()
                    mean = logits[:, :self.action_dim]
                    var = logits[:, self.action_dim:].exp()
                    cov = torch.diag_embed(var)
                    dist = MultivariateNormal(mean, cov)
                    dist_entropy = dist.entropy()
                    logprob = dist.log_prob(a[:, i])
                    ratio = torch.exp(logprob - oldlogprob[:, i])
                    ratio_list.append(ratio)  # the ratios need to be newly computed to have policy gradients
                surr1 = ratio_list[0].detach() * ratio_list[1] * advantage
                surr2 = torch.clamp((ratio_list[0].detach()) * ratio_list[1], 1 - self.eps_clip, 1 + self.eps_clip)
                policy_loss2 = torch.min(surr1, surr2).mean()
                infos[f'Nash policy loss player 2'] = policy_loss2

                loss = self.policy_loss_coeff * (policy_loss1 + policy_loss2) + 1.0 * (common_layer_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.all_params, self.max_grad_norm)
                self.optimizer.step()
                total_loss += loss.item()
        # print('loss :', policy_loss1.item(),  policy_loss2.item(), common_layer_loss.item())
        self.data = [[] for _ in range(self._num_channel)]

        return total_loss, infos