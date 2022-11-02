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

# torch.autograd.set_detect_anomaly(True)

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
                '\nPolicy networks: ', self.policies,
                '\nValue networks: ', self.values,
                '\nCommon layers: ', self.common_layers)

        policy_params, value_params, common_val_params, feature_net_param = [], [], [], []
        for f, p, v in zip(self.feature_nets, self.policies, self.values):
            feature_net_param += list(f.parameters())
            policy_params += list(p.parameters())
            value_params += list(v.parameters())

        common_val_params = list(self.common_layers.parameters())
        self.all_params = feature_net_param + policy_params + value_params + common_val_params
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
        done_lst = np.array(done_lst)
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
            done_mask_ = torch.flip(done_mask, dims=(0,))

            # need to prcess the samples, separate for agents
            if self.args.ram:
                s_ = s.view(s.shape[0], 2, -1)
                s_prime_ = s_prime.view(s_prime.shape[0], 2, -1)
            else:
                s_ = s  # shape: (batch, agents, envs, C, H, W)
                s_prime_ = s_prime

            for _ in range(self.K_epoch):
                loss = 0.0
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

                        vs = self.v(feature_x, i).squeeze(dim=-1)  # take the state for the specific agent
                        vs_prime = self.v(feature_x_prime, i).squeeze(dim=-1)
                        assert vs_prime.shape == done_mask.shape
                        r = r.detach()
                        vs_target = r[:, i] + self.gamma * vs_prime * done_mask
                        delta = vs_target - vs
                        advantage_lst = []
                        advantage = 0.0
                        for delta_t, mask in zip(torch.flip(delta, [-1]), done_mask_):  # reverse the delta along the time sequence in an episodic data
                            advantage = self.gamma * self.lmbda * advantage * mask + delta_t
                            advantage_lst.append(advantage)
                        advantage_lst.reverse()
                        advantage = torch.tensor(advantage_lst, dtype=torch.float).to(self.device)
                        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)  # this can have significant improvement (efficiency, stability) on performance
                        advantage = advantage.detach()
                        vs_target = advantage + vs

                    # value and policy loss for one agent
                    pi = self.pi(feature_x, i)
                    dist = Categorical(pi)
                    dist_entropy = dist.entropy()
                    logprob = dist.log_prob(a[:, i].squeeze())
                    ratio = torch.exp(logprob.squeeze() - oldlogprob[:, i].squeeze())
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
                    policy_loss = -torch.min(surr1, surr2)
                    v_loss = F.mse_loss(vs.squeeze(dim=-1), vs_target.detach())
                    ppo_loss = policy_loss + self.vf_coeff * v_loss - self.entropy_coeff * dist_entropy  # TODO vec + scalar + vec, is this valid?
                    ppo_loss = ppo_loss.mean()
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
                    for delta_t, mask in zip(torch.flip(delta, [-1]), done_mask_):  # reverse the delta along the time sequence in an episodic data
                        advantage = self.gamma * self.lmbda * advantage * mask + delta_t
                        advantage_lst.append(advantage)
                    advantage_lst.reverse()
                    advantage = torch.tensor(advantage_lst, dtype=torch.float).to(self.device)
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)  # this can have significant improvement (efficiency, stability) on performance
                    advantage = advantage.detach()
                    

                ratio_list = []
                for i in range(2):  # get the ratio for both
                    pi = self.pi(feature_x_list[i], i)
                    dist = Categorical(pi)
                    dist_entropy = dist.entropy()
                    logprob = dist.log_prob(a[:, i].squeeze())
                    ratio = torch.exp(logprob.squeeze() - oldlogprob[:, i].squeeze())
                    ratio_list.append(ratio)  # the ratios need to be newly computed to have policy gradients
                surr1 = ratio_list[0] * ratio_list[1].detach() * advantage
                surr2 = torch.clamp(ratio_list[0] * (ratio_list[1].detach()), 1 - self.eps_clip, 1 + self.eps_clip)
                policy_loss1 = -torch.min(surr1, surr2).mean()

                ratio_list = []
                for i in range(2):  # get the ratio for both
                    pi = self.pi(feature_x_list[i], i)
                    dist = Categorical(pi)
                    dist_entropy = dist.entropy()
                    logprob = dist.log_prob(a[:, i].squeeze())
                    ratio = torch.exp(logprob.squeeze() - oldlogprob[:, i].squeeze())
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
        self.num_agents = 2
        self.log_std_min = -20
        self.log_std_max = 2
        
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
        logprobs = []
        if Greedy:
            for policy, feature_net, state_per_agent in zip(self.policies, self.feature_nets, s):
                feature = feature_net(torch.from_numpy(np.array(state_per_agent)).unsqueeze(0).float().to(self.device))  # make sure input state shape is correct
                logits = policy(feature)

                if len(logits.shape) > 2:
                    logits = logits.squeeze()
                mean = torch.tanh(logits[:, :self.action_dim])
                log_std = logits[:, self.action_dim:]  # no tanh on log var
                log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)  # clipped to prevent blowing std
                std = log_std.exp()                
                actions.append(mean.detach().cpu().numpy())
            actions = np.array(actions)
            return actions
        else:
            for policy, feature_net, state_per_agent in zip(self.policies, self.feature_nets, s):
                feature = feature_net(torch.from_numpy(np.array(state_per_agent)).unsqueeze(0).float().to(self.device))
                logits = policy(feature)
                if len(logits.shape) > 2:
                    logits = logits.squeeze()
                mean = torch.tanh(logits[:, :self.action_dim])
                log_std = logits[:, self.action_dim:]  # no tanh on log var
                log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)  # clipped to prevent blowing std
                std = log_std.exp()
                # cov = torch.diag_embed(var)
                # dist = MultivariateNormal(mean, cov)
                # a = dist.sample()
                # logprob = dist.log_prob(a) 
      
                # normal = Normal(0, 1)
                # z      = normal.sample()
                # a = mean + std*z
                # logprob = Normal(mean, std).log_prob(a.squeeze())
                # logprob = logprob.sum(dim=-1, keepdim=True)  # reduce dim

                normal = Normal(mean, std)
                a = normal.sample()
                logprob = normal.log_prob(a).sum(-1)

                actions.append(a.detach().cpu().numpy())
                logprobs.append(logprob.detach().cpu().numpy())
            actions = np.array(actions)
            logprobs = np.array(logprobs)
            return actions, logprobs

    def get_log_prob(self, mean, std, action):
        log_prob = Normal(mean, std).log_prob(action)
        log_prob = log_prob.sum(dim=-1, keepdim=True)  # reduce dim
        return log_prob

    def get_action_log_prob(self, a, x, i):
        logits = self.pi(x, i)
        if len(logits.shape) > 2:
            logits = logits.squeeze()
        mean = torch.tanh(logits[:, :self.action_dim])
        log_std = logits[:, self.action_dim:]  # no tanh on log var
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)  # clipped to prevent blowing std
        std = log_std.exp()
        # cov = torch.diag_embed(var)
        # dist = MultivariateNormal(mean, cov)
        # dist_entropy = dist.entropy()
        # logprob = dist.log_prob(a[:, i].squeeze())  # for multivariate normal, sum of log_prob is produce of prob

        logprob = self.get_log_prob(mean, std, a[:, i].squeeze())
        dist_entropy = Normal(mean, std).entropy()
        dist_entropy = dist_entropy.sum(dim=-1, keepdim=True).mean()  # reduce dim
        return logprob, dist_entropy

    def update(self):
        infos = {}
        total_loss = 0.
        self.data = [x for x in self.data if x]  # remove empty

        s,a,r,s_prime,oldlogprob,done_mask = [],[],[],[],[],[]
        s = torch.tensor(s).to(self.device)
        a = torch.tensor(a).to(self.device)
        r = torch.tensor(r).to(self.device)
        s_prime = torch.tensor(s_prime).to(self.device)
        oldlogprob = torch.tensor(oldlogprob).to(self.device)
        done_mask = torch.tensor(done_mask).to(self.device)   # 0 if done

        for data in self.data:  # iterate over data from different environments
            traj_s, traj_a, traj_r, traj_s_prime, traj_oldlogprob, traj_done_mask = self.make_batch(data)
            s = torch.cat([s, traj_s])
            a = torch.cat([a, traj_a.view(traj_a.shape[0],2,-1)])
            r = torch.cat([r, traj_r])
            s_prime = torch.cat([s_prime, traj_s_prime])
            oldlogprob = torch.cat([oldlogprob, traj_oldlogprob])
            done_mask = torch.cat([done_mask, traj_done_mask])

            # need to prcess the samples, separate for agents
            if self.args.ram:
                s_ = s.view(s.shape[0], 2, -1)
                s_prime_ = s_prime.view(s_prime.shape[0], 2, -1)
            else:
                s_ = s  # shape: (batch, agents, envs, C, H, W)
                s_prime_ = s_prime   

        done_mask_ = torch.flip(done_mask, dims=(0,))

        # standard PPO
        for i in range(self.num_agents):  # for each agent
            # shared feature extraction
            if self.args.ram:
                feature_x = self.feature_nets[i](s_[:, i, :])
                feature_x_prime = self.feature_nets[i](s_prime_[:, i, :])
            else:
                feature_x = self.feature_nets[i](s_[:, i])
                feature_x_prime = self.feature_nets[i](s_prime_[:, i])    
            with torch.no_grad():                        
                vs = self.v(feature_x, i).squeeze(dim=-1)  # take the state for the specific agent
                advantage = torch.zeros_like(r[:, i])
                lastgaelam = 0
                for t in reversed(range(s_.shape[0])):
                    if not done_mask[t] or t == s_.shape[0]-1:   # 0 if done
                        nextvalues = self.v(feature_x_prime[t], i).squeeze()
                    else:
                        nextvalues = vs[t+1]     
                    # assert nextvalues.shape == vs[t].shape
                    delta = r[:, i][t] + self.gamma * nextvalues - vs[t]
                    advantage[t] = lastgaelam = delta + self.gamma * self.lmbda * lastgaelam

                assert advantage.shape == vs.shape
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                vs_target = advantage + vs
                
 
                # vs_prime = self.v(feature_x_prime, i).squeeze(dim=-1)
                # assert vs_prime.shape == done_mask.shape
                # r = r.detach()
                # vs_target = r[:, i] + self.gamma * vs_prime * done_mask
                # delta = vs_target - vs
                # advantage_lst = []
                # advantage = 0.0
                # for delta_t, mask in zip(torch.flip(delta, [-1]), done_mask_):  # reverse the delta along the time sequence in an episodic data
                #     advantage = self.gamma * self.lmbda * advantage * mask + delta_t
                #     advantage_lst.append(advantage)
                # advantage_lst.reverse()
                # advantage = torch.tensor(advantage_lst, dtype=torch.float).to(self.device)
                # advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)  # this can have significant improvement (efficiency, stability) on performance
                # advantage = advantage.detach()
                # vs_target = advantage + vs

        ratios = [[] for _ in range(self.num_agents)]
        values = [[] for _ in range(self.num_agents)]
        stds = [[] for _ in range(self.num_agents)]
        ppo_total_loss = [0. for _ in range(self.num_agents)]
        p_loss = [0. for _ in range(self.num_agents)]
        v_loss = [0. for _ in range(self.num_agents)]
        nash_v_loss = 0.
        nash_policy_loss = [0. for _ in range(self.num_agents)]

        dist_entropies = [[] for _ in range(self.num_agents)]

        for _ in range(self.K_epoch):
            loss = 0.0
            # Standard PPO
            for i in range(self.num_agents):  # for each agent
                if self.args.ram:
                    feature_x = self.feature_nets[i](s_[:, i, :])
                    feature_x_prime = self.feature_nets[i](s_prime_[:, i, :])
                else:
                    feature_x = self.feature_nets[i](s_[:, i])
                    feature_x_prime = self.feature_nets[i](s_prime_[:, i])                        

                new_vs = self.v(feature_x, i)  # take the state for the specific agent
                logprob, dist_entropy = self.get_action_log_prob(a, feature_x, i)
 
                ratio = torch.exp(logprob.squeeze() - oldlogprob[:, i].squeeze())
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
                policy_loss = -torch.min(surr1, surr2).mean()
                # value_loss = F.mse_loss(new_vs.squeeze(dim=-1), vs_target.detach())

                # clipped value loss
                v_clipped = vs + torch.clamp(new_vs - vs, -self.eps_clip, self.eps_clip)
                value_loss_clipped = (v_clipped - vs_target.detach()) ** 2
                value_loss_unclipped = (new_vs - vs_target.detach()) ** 2
                value_loss_max = torch.max(value_loss_unclipped, value_loss_clipped)
                value_loss =  0.5 * value_loss_max.mean()

                ppo_loss = policy_loss + self.vf_coeff * value_loss - self.entropy_coeff * dist_entropy  # TODO vec + scalar + vec, is this valid?
                ppo_loss = ppo_loss.mean()

                ppo_total_loss[i] += ppo_loss.item()
                p_loss[i] += policy_loss.item()
                v_loss[i] += value_loss.item()
                dist_entropies[i].append(dist_entropy.item())
                ratios[i].append(ratio.mean().item())
                values[i].append(new_vs.mean().item())
                # stds[i].append(std.mean().item())

                self.optimizer.zero_grad()
                ppo_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.all_params, self.max_grad_norm)
                self.optimizer.step()
                total_loss += ppo_loss.item()
                
            # loss for common layers (value function)
            # common layers
            feature_x_list = []
            feature_x_prime_list = []
            for i in range(2):  # for each agent
                feature_x = self.feature_nets[i](s_[:, i, :])
                feature_x_prime = self.feature_nets[i](s_prime_[:, i, :])
                feature_x_list.append(feature_x)
                feature_x_prime_list.append(feature_x_prime)     

            with torch.no_grad():   
                common_vs = self.common_layers(torch.cat(feature_x_list, axis=1)).squeeze(dim=-1)  # TODO just use the first state (assume it has full info)
                advantage = torch.zeros_like(r[:, 0])
                lastgaelam = 0
                for t in reversed(range(s_.shape[0])):
                    if not done_mask[t] or t == s_.shape[0]-1:   # 0 if done
                        nextvalues = self.common_layers(torch.cat(feature_x_prime_list, axis=1)[t]).squeeze()
                    else:
                        nextvalues = common_vs[t+1]   
                    # assert nextvalues.shape == common_vs[t].shape
                    delta = r[:, 0][t] + self.gamma * nextvalues - common_vs[t]
                    advantage[t] = lastgaelam = delta + self.gamma * self.lmbda * lastgaelam

                assert advantage.shape == common_vs.shape
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                common_vs_target = advantage + common_vs

                # vs_prime = self.common_layers(torch.cat(feature_x_prime_list, axis=1)).squeeze(dim=-1)  # TODO just use the first state (assume it has full info)
                # assert vs_prime.shape == done_mask.shape
                # common_vs = self.common_layers(torch.cat(feature_x_list, axis=1)).squeeze(dim=-1)  # TODO just use the first state (assume it has full info)
                # common_vs_target = r[:, 0] + self.gamma * vs_prime * done_mask  # r is the first player's here
                # # calculate generalized advantage with common layer value
                # delta = common_vs_target - common_vs
                # delta = delta.detach()
                # advantage_lst = []
                # advantage = 0.0
                # for delta_t, mask in zip(torch.flip(delta, [-1]), done_mask_):  # reverse the delta along the time sequence in an episodic data
                #     advantage = self.gamma * self.lmbda * advantage * mask + delta_t
                #     advantage_lst.append(advantage)
                # advantage_lst.reverse()
                # advantage = torch.tensor(advantage_lst, dtype=torch.float).to(self.device)
                # advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)  # this can have significant improvement (efficiency, stability) on performance
                # advantage = advantage.detach()
                # common_vs_target = advantage + common_vs

            new_common_vs = self.common_layers(torch.cat(feature_x_list, axis=1))  # TODO just use the first state (assume it has full info)
            common_layer_loss = F.mse_loss(new_common_vs.squeeze(dim=-1), common_vs_target.detach()).mean()
            nash_v_loss += common_layer_loss.item()
            ratio_list = []
            for i in range(2):  # get the ratio for both
                logprob, _ = self.get_action_log_prob(a, feature_x_list[i], i)
                ratio = torch.exp(logprob.squeeze() - oldlogprob[:, i].squeeze())
                ratio_list.append(ratio)  # the ratios need to be newly computed to have policy gradients
            surr1 = ratio_list[0] * ratio_list[1].detach() * advantage
            surr2 = torch.clamp(ratio_list[0] * (ratio_list[1].detach()), 1 - self.eps_clip, 1 + self.eps_clip)
            policy_loss1 = -torch.min(surr1, surr2).mean()
            nash_policy_loss[0] += policy_loss1.item()

            ratio_list = []
            for i in range(2):  # get the ratio for both
                logprob, _ = self.get_action_log_prob(a, feature_x_list[i], i)
                ratio = torch.exp(logprob.squeeze() - oldlogprob[:, i].squeeze())
                ratio_list.append(ratio)  # the ratios need to be newly computed to have policy gradients
            surr1 = ratio_list[0].detach() * ratio_list[1] * advantage
            surr2 = torch.clamp((ratio_list[0].detach()) * ratio_list[1], 1 - self.eps_clip, 1 + self.eps_clip)
            policy_loss2 = torch.min(surr1, surr2).mean()
            nash_policy_loss[1] += policy_loss2.item()

            loss = self.policy_loss_coeff * (policy_loss1 + policy_loss2) + 1.0 * (common_layer_loss)

            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.all_params, self.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()
        # print('loss :', policy_loss1.item(),  policy_loss2.item(), common_layer_loss.item())

        infos[f'PPO policy loss player {i}'] = p_loss[i]
        infos[f'PPO value loss player {i}'] = v_loss[i]
        infos[f'PPO total loss player {i}'] = ppo_total_loss[i]
        infos[f'policy entropy player {i}'] = np.mean(dist_entropies[i])
        # infos[f'PPO policy std player {i}'] = np.mean(stds[i])
        infos[f'PPO policy ratio player {i}'] = np.mean(ratios[i])
        infos[f'PPO mean_value player {i}'] = np.mean(values[i])
        infos[f'Nash value loss'] = nash_v_loss
        infos[f'Nash policy loss player 1'] = nash_policy_loss[0]
        infos[f'Nash policy loss player 2'] = nash_policy_loss[1]

        self.data = [[] for _ in range(self._num_channel)]

        return total_loss, infos