from cmath import log
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal, MultivariateNormal
import numpy as np
import gym
from .agent import Agent
from ..common.networks import MLP, CNN, get_model
from ..common.rl_utils import choose_optimizer
from mars.utils.typing import List, Tuple, StateType, ActionType, SampleType, SingleEnvMultiAgentSampleType

def PPO(env, args):
    """ The function returns a proper class for Proximal Policy Optimization (PPO) algorithm,
    according to the action space of the environment.

    :param env: environment instance
    :type env: object
    :param args: arguments
    :type args: ConfigurationDict
    """    
    if isinstance(env.action_space, list):
        if isinstance(env.action_space[0], gym.spaces.Box):
            return PPOContinuous(env, args)
        else:
            return PPODiscrete(env, args)
    else:
        if isinstance(env.action_space, gym.spaces.Box):
            return PPOContinuous(env, args)
        else:
            return PPODiscrete(env, args)

class PPOBase(Agent):
    """ PPO agorithm for environments with continuous action space.
    """ 
    def __init__(self, env, args):
        super().__init__(env, args)
        self.learning_rate = args.learning_rate
        self.gamma = float(args.algorithm_spec['gamma'])
        self.lmbda = float(args.algorithm_spec['lambda'])
        self.eps_clip = float(args.algorithm_spec['eps_clip'])
        self.K_epoch = args.algorithm_spec['K_epoch']
        self.GAE = args.algorithm_spec['GAE']
        self.max_grad_norm = float(args.algorithm_spec['max_grad_norm'])
        self.ini_entropy_coeff = float(args.algorithm_spec['entropy_coeff'])
        self.entropy_coeff = self.ini_entropy_coeff
        self.vf_coeff = float(args.algorithm_spec['vf_coeff'])
        self.policy_logstd = None # only exist for continuous
        self._init_model(env, args)
        self.optim_parameters = list(self.feature.parameters())+list(self.value.parameters())+list(self.policy.parameters())
        if self.policy_logstd is not None:
            # self.optim_parameters.append(self.policy_logstd)
            self.optim_parameters += list(self.policy_logstd.parameters())
        self.optimizer = choose_optimizer(args.optimizer)(self.optim_parameters, lr=float(args.learning_rate))
        self.mseLoss = nn.MSELoss()
        self._num_channel = args.num_envs*(env.num_agents if isinstance(env.num_agents, int) else env.num_agents[0]) # env.num_agents is a list when using parallel envs 
        self.data = [[] for _ in range(self._num_channel)]

    def _init_model(self, env, args, policy_type=None):
        if policy_type is not None:
            self.policy_type = policy_type
        if len(self.observation_space.shape) <= 1:
            self.feature_space = self.observation_space
            self.feature = MLP(env.observation_space, self.feature_space, args.net_architecture['feature'], model_for='feature').to(self.device)
            self.policy = MLP(self.feature_space, env.action_space, args.net_architecture['policy'], model_for=self.policy_type).to(self.device)
            self.value = MLP(self.feature_space, env.action_space, args.net_architecture['value'], model_for='value').to(self.device)

        else:
            self.feature_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape = (256,))
            self.feature = CNN(env.observation_space, self.feature_space, args.net_architecture['feature'], model_for='feature').to(self.device)
            self.policy = MLP(self.feature_space, env.action_space, args.net_architecture['policy'], model_for=self.policy_type).to(self.device)
            self.value = MLP(self.feature_space, env.action_space, args.net_architecture['value'], model_for='value').to(self.device)
        
        if args.num_process > 1:
            self.feature.share_memory()
            self.policy.share_memory()
            self.value.share_memory()  


    def pi(
        self, 
        x: List[StateType]
        ) -> List[ActionType]:
        """ Forward the policy network.

        :param x: input of the policy network, i.e. the state
        :type x: List[StateType]
        :return: the logits/actions
        :rtype: List[ActionType]
        """ 
        pass

    def v(
        self, 
        x: List[StateType]
        ) -> List[float]:
        """ Forward the value network.

        :param x: input of the value network, i.e. the state
        :type x: List[StateType]
        :return: a list of values for each state
        :rtype: List[float]
        """    
        feature = self.feature(x)    
        return self.value.forward(feature)   
    
    def reinit(self,):
        self.policy.reinit()
        self.value.reinit()

    def store(self, transitions: SampleType) -> None:
        """ Store samples in batch.

        :param transitions: a list of samples from different environments (if using parallel env)
        :type transitions: SampleType
        """        
        # If several transitions are pushed at the same time,
        # they are not from the same trajectory, therefore they need
        # to be stored separately since PPO is on-policy.
        for i, transition in enumerate(transitions): # iterate over the list
            self.data[i].append(transition)

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
            a_lst.append(a.squeeze())
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            prob_a_lst.append(prob_a.squeeze())
            done_mask = 0 if done else 1
            done_lst.append(done_mask)
        a_lst = np.array(a_lst)
        r_lst = np.array(r_lst)
        done_mask = np.array(done_mask)
        prob_a_lst = np.array(prob_a_lst)
        # found this step take some time for Pong (not ram), even if no parallel no multiagent
        s,a,r,s_prime,prob_a,done_mask =    torch.tensor(s_lst, dtype=torch.float).to(self.device), torch.tensor(a_lst).to(self.device), \
                                            torch.tensor(r_lst, dtype=torch.float).to(self.device), torch.tensor(s_prime_lst, dtype=torch.float).to(self.device), \
                                            torch.tensor(prob_a_lst, dtype=torch.float).to(self.device), torch.tensor(done_lst, dtype=torch.float).to(self.device)
        return s, a, r, s_prime, prob_a, done_mask
        
    def update(self):
        pass

    def save_model(self, path=None):
        try:  # for PyTorch >= 1.7 to be compatible with loading models from any lower version
            torch.save(self.feature.state_dict(), path+'_feature', _use_new_zipfile_serialization=False)
            torch.save(self.policy.state_dict(), path+'_policy', _use_new_zipfile_serialization=False)
            if self.policy_logstd is not None:
                torch.save(self.policy_logstd, path+'_policy_logstd', _use_new_zipfile_serialization=False)
            torch.save(self.value.state_dict(), path+'_value', _use_new_zipfile_serialization=False)
        except:
            torch.save(self.feature.state_dict(), path+'_feature')
            torch.save(self.policy.state_dict(), path+'_policy')
            if self.policy_logstd is not None:
                torch.save(self.policy_logstd, path+'_policy_logstd')
            torch.save(self.value.state_dict(), path+'_value')


    def load_model(self, path=None):
        self.feature.load_state_dict(torch.load(path+'_feature'))
        self.policy.load_state_dict(torch.load(path+'_policy'))
        if self.policy_logstd is not None:
            self.policy_logstd = torch.load(path+'_policy_logstd')
        self.value.load_state_dict(torch.load(path+'_value'))


class PPODiscrete(PPOBase):
    """ PPO agorithm for environments with discrete action space.
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
        prob = self.pi(torch.from_numpy(s).unsqueeze(0).float().to(self.device)).squeeze()  # make sure input state shape is correct
        if Greedy:
            a = torch.argmax(prob, dim=-1).detach().cpu().numpy()
            return a
        else:
            dist = Categorical(prob)
            a = dist.sample()
            logprob = dist.log_prob(a)
            return a.detach().cpu().numpy(), logprob.detach().cpu().numpy()

    def pi(
        self, 
        x: List[StateType]
        ) -> List[ActionType]:
        """ Forward the policy network.

        :param x: input of the policy network, i.e. the state
        :type x: List[StateType]
        :return: the logits/actions
        :rtype: List[ActionType]
        """ 
        feature = self.feature(x)
        return self.policy.forward(feature)


    # def update(self):
    #     infos = {}
    #     total_loss, p_loss, v_loss = 0., 0., 0.
    #     self.data = [x for x in self.data if x]  # remove empty
    #     for data in self.data: # iterate over data from different environments
    #         s, a, r, s_prime, oldlogprob, done_mask = self.make_batch(data)
    #         done_mask_ = torch.flip(done_mask, dims=(0,))
    #         if not self.GAE:
    #             rewards = []
    #             discounted_r = 0
    #             for reward, is_continue in zip(reversed(r), reversed(done_mask)):
    #                 if not is_continue:
    #                     discounted_r = 0
    #                 discounted_r = reward + self.gamma * discounted_r
    #                 rewards.insert(0, discounted_r)  # insert in front, cannot use append

    #             ## reward normalization is not common: by Costa Huang
    #             # rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
    #             # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            
    #         for _ in range(self.K_epoch):
    #             vs = self.v(s)

    #             if self.GAE:
    #                 # use generalized advantage estimation
    #                 vs_prime = self.v(s_prime).squeeze(dim=-1)
    #                 assert vs_prime.shape == done_mask.shape
    #                 vs_target = r + self.gamma * vs_prime * done_mask
    #                 delta = vs_target - vs.squeeze(dim=-1)
    #                 delta = delta.detach()
    #                 advantage_lst = []
    #                 advantage = 0.0
    #                 for delta_t, mask in zip(torch.flip(delta, [-1]), done_mask_): # reverse the delta along the time sequence in an episodic data
    #                     advantage = self.gamma * self.lmbda * advantage * mask + delta_t
    #                     advantage_lst.append(advantage)
    #                 advantage_lst.reverse()
    #                 advantage = torch.tensor(advantage_lst, dtype=torch.float).to(self.device)
    #                 advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)  # this can have significant improvement (efficiency, stability) on performance

    #             else:
    #                 advantage = rewards - vs.squeeze(dim=-1).detach()
    #                 vs_target = rewards

    #             pi = self.pi(s)
    #             dist = Categorical(pi)
    #             dist_entropy = dist.entropy()
    #             logprob = dist.log_prob(a)
    #             ratio = torch.exp(logprob - oldlogprob)
    #             surr1 = ratio * advantage
    #             surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage
    #             policy_loss = -torch.min(surr1, surr2).mean()
    #             value_loss = self.mseLoss(vs.squeeze(dim=-1) , vs_target.detach())
    #             loss = policy_loss + self.vf_coeff*value_loss - self.entropy_coeff*dist_entropy

    #             self.optimizer.zero_grad()
    #             mean_loss = loss.mean()
    #             mean_loss.backward()
    #             nn.utils.clip_grad_norm_(list(self.value.parameters())+list(self.policy.parameters()), self.max_grad_norm)
    #             self.optimizer.step()

    #             total_loss += mean_loss.item()
    #             p_loss += policy_loss.item()
    #             v_loss += value_loss.item()

    #     infos[f'PPO policy loss'] = p_loss
    #     infos[f'PPO value loss'] = v_loss
    #     infos[f'PPO total loss'] = total_loss
    #     infos[f'policy entropy'] = dist_entropy

    #     self.data = [[] for _ in range(self._num_channel)]

    #     return total_loss, infos
 
        
    def update(self):

        infos = {}
        total_loss, p_loss, v_loss = 0., 0., 0.
        self.data = [x for x in self.data if x]  # remove empty
        
        s,a,r,s_prime,oldlogprob,done_mask = [],[],[],[],[],[]
        s = torch.tensor(s).to(self.device)
        a = torch.tensor(a).to(self.device)
        r = torch.tensor(r).to(self.device)
        s_prime = torch.tensor(s_prime).to(self.device)
        oldlogprob = torch.tensor(oldlogprob).to(self.device)
        done_mask = torch.tensor(done_mask).to(self.device)
    
        for data in self.data: # iterate over data from different environments
            traj_s, traj_a, traj_r, traj_s_prime, traj_oldlogprob, traj_done_mask = self.make_batch(data)
            s = torch.cat([s, traj_s])
            a = torch.cat([a, traj_a])
            r = torch.cat([r, traj_r])
            s_prime = torch.cat([s_prime, traj_s_prime])
            oldlogprob = torch.cat([oldlogprob, traj_oldlogprob])
            done_mask = torch.cat([done_mask, traj_done_mask])
        
        done_mask_ = torch.flip(done_mask, dims=(0,))

        # the target value calculation should be outside epochs of update (more stable)
        vs = self.v(s)
        vs_prime = self.v(s_prime).squeeze(dim=-1)
        if self.GAE:
            # use generalized advantage estimation
            assert vs_prime.shape == done_mask.shape
            vs_target = r + self.gamma * vs_prime * done_mask
            delta = vs_target - vs.squeeze(dim=-1)
            delta = delta.detach()
            advantage_lst = []
            advantage = 0.0
            for delta_t, mask in zip(torch.flip(delta, [-1]), done_mask_): # reverse the delta along the time sequence in an episodic data
                advantage = self.gamma * self.lmbda * advantage * mask + delta_t
                advantage_lst.append(advantage)
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float).to(self.device)

        else:
            rewards = []
            discounted_r = 0
            for reward, is_continue in zip(reversed(r), done_mask_):
                if not is_continue:
                    discounted_r = 0
                discounted_r = reward + self.gamma * discounted_r
                rewards.insert(0, discounted_r)  # insert in front, cannot use append
            rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
            advantage = rewards - vs.squeeze(dim=-1).detach()
            vs_target = rewards

        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        for _ in range(self.K_epoch):
            new_vs = self.v(s)
            pi = self.pi(s)
            dist = Categorical(pi)
            dist_entropy = dist.entropy().mean()
            logprob = dist.log_prob(a)
            ratio = torch.exp(logprob - oldlogprob)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = self.mseLoss(new_vs.squeeze(dim=-1) , vs_target.detach())
            loss = policy_loss + self.vf_coeff*value_loss - self.entropy_coeff*dist_entropy

            self.optimizer.zero_grad()
            mean_loss = loss.mean()
            mean_loss.backward()
            nn.utils.clip_grad_norm_(self.optim_parameters, self.max_grad_norm)
            self.optimizer.step()

            total_loss += mean_loss.item()
            p_loss += policy_loss.item()
            v_loss += value_loss.item()

        infos[f'PPO policy loss'] = p_loss
        infos[f'PPO value loss'] = v_loss
        infos[f'PPO total loss'] = total_loss
        infos[f'policy entropy'] = dist_entropy

        self.data = [[] for _ in range(self._num_channel)]

        return total_loss, infos

class PPOContinuous(PPOBase):
    """ PPO agorithm for environments with continuous action space.
    """ 
    def __init__(self, env, args):
        super().__init__(env, args)
        # target entropy set according to SAC: https://arxiv.org/pdf/1812.05905.pdf
        try:
            self.target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(self.device)).detach()
        except:
            self.target_entropy = -torch.prod(torch.Tensor(env.action_space[0].shape).to(self.device)).detach()
        self.log_std_min = -20
        self.log_std_max = 2
        # self.log_entropy_coef = torch.zeros(1, requires_grad=True, device=self.device)
        # self.entropy_coeff = self.log_entropy_coef.exp().item()
        # self.coef_optimizer = optim.Adam([self.log_entropy_coef], lr=float(args.learning_rate))

    def _init_model(self, env, args):
        super()._init_model(env, args, policy_type='independent_gaussian_policy')
        # action_space = env.action_space[0] if isinstance(env.action_space, list) else env.action_space
        # self.policy_logstd = nn.Parameter(torch.zeros((1, np.prod(action_space.shape)), requires_grad=True, device=self.device))
        
        # state-specific std is found to perform much better than parameter only
        self.policy_logstd = MLP(self.feature_space, env.action_space, args.net_architecture['policy'], model_for='independent_gaussian_policy').to(self.device)
        if args.num_process > 1:
            self.policy_logstd.share_memory()

    def pi(
        self, 
        x: List[StateType]
        ) -> List[ActionType]:
        """ Forward the policy network.

        :param x: input of the policy network, i.e. the state
        :type x: List[StateType]
        :return: the logits/actions
        :rtype: List[ActionType]
        """ 
        feature = self.feature(x)
        policy_mean = self.policy.forward(feature)
        policy_mean = torch.tanh(policy_mean)
        if len(policy_mean.shape) > 2:
            policy_mean = policy_mean.squeeze()
        # policy_logstd = self.policy_logstd.expand_as(policy_mean)
        policy_logstd = self.policy_logstd(feature.detach())
        if len(policy_logstd.shape) > 2:
            policy_logstd = policy_logstd.squeeze()
        return policy_mean, policy_logstd

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
        mean, policy_logstd = self.pi(torch.from_numpy(s).unsqueeze(0).float().to(self.device))  # make sure input state shape is correct
        log_std = torch.clamp(policy_logstd, self.log_std_min, self.log_std_max)  # clipped to prevent blowing std
        std = log_std.exp()

        if Greedy:
            a = mean.detach().cpu().numpy()
            return a
        else:
            # cov = torch.diag_embed(var)
            # dist = MultivariateNormal(mean, cov)
            # a = dist.sample()
            # logprob = dist.log_prob(a)
            
            # normal = Normal(0, 1)
            # z      = normal.sample()
            # a = mean + std*z
            # logprob = Normal(mean, std).log_prob(a)
            # logprob = logprob.sum(dim=-1, keepdim=True)  # reduce dim
            normal = Normal(mean, std)
            a = normal.sample()
            logprob = normal.log_prob(a).sum(-1)
            return a.detach().cpu().numpy(), logprob.detach().cpu().numpy()

    def get_log_prob(self, mean, std, action):
        log_prob = Normal(mean, std).log_prob(action)
        log_prob = log_prob.sum(dim=-1, keepdim=True)  # reduce dim
        return log_prob

    def update(self):
        infos = {}
        total_loss, p_loss, v_loss = 0., 0., 0.
        ratios, values, stds = [], [], []
        self.data = [x for x in self.data if x]  # remove empty
        
        s,a,r,s_prime,oldlogprob,done_mask = [],[],[],[],[],[]
        s = torch.tensor(s).to(self.device)
        a = torch.tensor(a).to(self.device)
        r = torch.tensor(r).to(self.device)
        s_prime = torch.tensor(s_prime).to(self.device)
        oldlogprob = torch.tensor(oldlogprob).to(self.device)
        done_mask = torch.tensor(done_mask).to(self.device)

        for data in self.data: # iterate over data from different environments
            traj_s, traj_a, traj_r, traj_s_prime, traj_oldlogprob, traj_done_mask = self.make_batch(data)
            s = torch.cat([s, traj_s])
            a = torch.cat([a, traj_a])
            r = torch.cat([r, traj_r])
            s_prime = torch.cat([s_prime, traj_s_prime])
            oldlogprob = torch.cat([oldlogprob, traj_oldlogprob])
            done_mask = torch.cat([done_mask, traj_done_mask])
        done_mask_ = torch.flip(done_mask, dims=(0,))

        # the target value calculation should be outside epochs of update (more stable)
        vs = self.v(s)
        vs_prime = self.v(s_prime).squeeze(dim=-1)
        if self.GAE:
            # use generalized advantage estimation
            assert vs_prime.shape == done_mask.shape
            delta = r + self.gamma * vs_prime * done_mask - vs.squeeze(dim=-1)
            delta = delta.detach()
            advantage_lst = []
            advantage = 0.0
            for delta_t, mask in zip(torch.flip(delta, [-1]), done_mask_): # reverse the delta along the time sequence in an episodic data
                advantage = self.gamma * self.lmbda * advantage * mask + delta_t
                advantage_lst.append(advantage)
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float).to(self.device)
            vs_target = advantage + vs.squeeze(dim=-1) 

        else:
            rewards = []
            discounted_r = 0
            for reward, is_continue in zip(reversed(r), done_mask_):
                if not is_continue:
                    discounted_r = 0
                discounted_r = reward + self.gamma * discounted_r
                rewards.insert(0, discounted_r)  # insert in front, cannot use append
            rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
            advantage = rewards - vs.squeeze(dim=-1).detach()
            vs_target = rewards

        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        for _ in range(self.K_epoch):
            new_vs = self.v(s)

            mean, policy_logstd = self.pi(s)
            log_std = torch.clamp(policy_logstd, self.log_std_min, self.log_std_max)  # clipped to prevent blowing std
            std = log_std.exp()            
            
            # cov = torch.diag_embed(var)
            # dist = MultivariateNormal(mean, cov)
            # dist_entropy = dist.entropy()
            # logprob = dist.log_prob(a)

            logprob = self.get_log_prob(mean, std, a.squeeze())
            dist_entropy = Normal(mean, std).entropy()
            dist_entropy = dist_entropy.sum(dim=-1, keepdim=True).mean()  # reduce dim

            ratio = torch.exp(logprob.squeeze() - oldlogprob.squeeze())  # squeeze is important to keep dim
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = self.mseLoss(new_vs.squeeze(dim=-1) , vs_target.detach())
            loss = policy_loss + self.vf_coeff*value_loss - self.entropy_coeff*dist_entropy
            mean_loss = loss.mean()

            total_loss += mean_loss.item()
            p_loss += policy_loss.item()
            v_loss += value_loss.item()

            self.optimizer.zero_grad()
            mean_loss.backward()
            nn.utils.clip_grad_norm_(self.optim_parameters, self.max_grad_norm)
            self.optimizer.step()

            # print(self.entropy_coeff, dist_entropy)
            # coef_loss = - (self.log_entropy_coef * (logprob + self.target_entropy).detach()).mean() # SAC auto entropy coeff loss
            # self.coef_optimizer.zero_grad()
            # coef_loss.backward()
            # self.coef_optimizer.step()
            
            # avoid entropy blowing up
            # if dist_entropy.mean() > self.target_entropy + 20.:
            #     self.entropy_coeff = -self.ini_entropy_coeff
            # else:
            #     self.entropy_coeff = self.ini_entropy_coeff

            ratios.append(ratio.mean().item())
            values.append(new_vs.mean().item())
            stds.append(std.mean().item())

        infos[f'PPO policy loss'] = p_loss
        infos[f'PPO value loss'] = v_loss
        infos[f'PPO total loss'] = total_loss
        infos[f'policy entropy'] = dist_entropy
        infos[f'policy std'] = np.mean(stds)
        infos[f'policy ratio'] = np.mean(ratios)
        infos[f'mean_value'] = np.mean(values)
        infos[f'entropy_coeff'] = self.entropy_coeff
        self.data = [[] for _ in range(self._num_channel)]

        return total_loss, infos