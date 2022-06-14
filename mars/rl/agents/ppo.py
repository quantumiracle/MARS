import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
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
    if isinstance(env.action_space, gym.spaces.Box) or isinstance(env.action_space[0], gym.spaces.Box): # discrete TODO
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
        self._init_model(env, args)

        if args.num_process > 1:
            self.policy.share_memory()
            self.policy_old.share_memory()
            self.value.share_memory()  

        self.optimizer = choose_optimizer(args.optimizer)(list(self.value.parameters())+list(self.policy.parameters()), lr=float(args.learning_rate))
        self.mseLoss = nn.MSELoss()
        self._num_channel = args.num_envs*(env.num_agents if isinstance(env.num_agents, int) else env.num_agents[0]) # env.num_agents is a list when using parallel envs 
        self.data = [[] for _ in range(self._num_channel)]

    def _init_model(self, env, args):

        if len(self.observation_space.shape) <= 1:
            self.policy = MLP(env.observation_space, env.action_space, args.net_architecture['policy'], model_for=self.policy_type).to(self.device)
            self.policy_old = copy.deepcopy(self.policy).to(self.device)
            # self.policy_old.load_state_dict(self.policy.state_dict())
            self.value = MLP(env.observation_space, env.action_space, args.net_architecture['value'], model_for='value').to(self.device)

        else:
            self.policy = CNN(env.observation_space, env.action_space, args.net_architecture['policy'], model_for=self.policy_type).to(self.device)
            self.policy_old = copy.deepcopy(self.policy).to(self.device)
            # self.policy_old.load_state_dict(self.policy.state_dict())
            self.value = CNN(env.observation_space, env.action_space, args.net_architecture['value'], model_for='value').to(self.device)

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
        return self.policy.forward(x)

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
        return self.value.forward(x)  
    
    def reinit(self,):
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
            a_lst.append(a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            prob_a_lst.append(prob_a)
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
            torch.save(self.policy.state_dict(), path+'_policy', _use_new_zipfile_serialization=False)
            torch.save(self.value.state_dict(), path+'_value', _use_new_zipfile_serialization=False)
        except:
            torch.save(self.policy.state_dict(), path+'_policy')
            torch.save(self.value.state_dict(), path+'_value')


    def load_model(self, path=None):
        self.policy.load_state_dict(torch.load(path+'_policy'))
        self.policy_old.load_state_dict(self.policy.state_dict())  # important

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
        prob = self.policy(torch.from_numpy(s).unsqueeze(0).float().to(self.device)).squeeze()  # make sure input state shape is correct
        if Greedy:
            a = torch.argmax(prob, dim=-1).detach().cpu().numpy()
            return a
        else:
            dist = Categorical(prob)
            a = dist.sample()
            logprob = dist.log_prob(a)
            return a.detach().cpu().numpy(), logprob.detach().cpu().numpy()
        
    def update(self):
        total_loss = 0.
        self.data = [x for x in self.data if x]  # remove empty
        for data in self.data: # iterate over data from different environments
            s, a, r, s_prime, oldlogprob, done_mask = self.make_batch(data)
            if not self.GAE:
                rewards = []
                discounted_r = 0
                for reward, is_continue in zip(reversed(r), reversed(done_mask)):
                    if not is_continue:
                        discounted_r = 0
                    discounted_r = reward + self.gamma * discounted_r
                    rewards.insert(0, discounted_r)  # insert in front, cannot use append

                rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            
            for _ in range(self.K_epoch):
                vs = self.v(s)

                if self.GAE:
                    # use generalized advantage estimation
                    vs_prime = self.v(s_prime).squeeze(dim=-1)
                    assert vs_prime.shape == done_mask.shape
                    vs_target = r + self.gamma * vs_prime * done_mask
                    delta = vs_target - vs.squeeze(dim=-1)
                    delta = delta.detach()
                    advantage_lst = []
                    advantage = 0.0
                    for delta_t in torch.flip(delta, [-1]): # reverse the delta along the time sequence in an episodic data
                        advantage = self.gamma * self.lmbda * advantage + delta_t
                        advantage_lst.append(advantage)
                    advantage_lst.reverse()
                    advantage = torch.tensor(advantage_lst, dtype=torch.float).to(self.device)
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)  # this can have significant improvement (efficiency, stability) on performance

                else:
                    advantage = rewards - vs.squeeze(dim=-1).detach()
                    vs_target = rewards

                pi = self.pi(s)
                dist = Categorical(pi)
                dist_entropy = dist.entropy()
                logprob = dist.log_prob(a)
                # pi_a = pi.gather(1,a)
                # ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))
                ratio = torch.exp(logprob - oldlogprob)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage
                loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(vs.squeeze(dim=-1) , vs_target.detach()) - 0.01*dist_entropy
                # loss = -torch.min(surr1, surr2) + 0.5*self.mseLoss(vs.squeeze(dim=-1) , vs_target.detach()) - 0.01*dist_entropy
                
                # print('vs: ', vs.shape, vs)
                # print('logprob', logprob.shape, logprob)
                # print('oldlogprob', oldlogprob.shape, oldlogprob)
                # print('advantage: ', advantage.shape, advantage)
                # print('loss', loss)

                self.optimizer.zero_grad()
                mean_loss = loss.mean()
                mean_loss.backward()
                self.optimizer.step()

                total_loss += mean_loss.item()

        self.data = [[] for _ in range(self._num_channel)]

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        return total_loss

class PPOContinuous(PPOBase):
    """ PPO agorithm for environments with continuous action space.
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
        logits = self.policy(torch.from_numpy(s).unsqueeze(0).float().to(self.device))  # make sure input state shape is correct
        if len(logits.shape) > 2:
            logits = logits.squeeze()
        mean = logits[:, :self.action_dim]
        std = logits[:, self.action_dim:].exp()

        if Greedy:
            a = mean.detach().cpu().numpy()
            return a
        else:
            dist = Normal(mean, std)
            a = dist.sample()
            logprob = dist.log_prob(a)
            return a.detach().cpu().numpy(), logprob.detach().cpu().numpy()

    def update(self):
        total_loss = 0.
        self.data = [x for x in self.data if x]  # remove empty
        for data in self.data: # iterate over data from different environments
            s, a, r, s_prime, oldlogprob, done_mask = self.make_batch(data)
            if not self.GAE:
                rewards = []
                discounted_r = 0
                for reward, is_continue in zip(reversed(r), reversed(done_mask)):
                    if not is_continue:
                        discounted_r = 0
                    discounted_r = reward + self.gamma * discounted_r
                    rewards.insert(0, discounted_r)  # insert in front, cannot use append

                rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            
            for _ in range(self.K_epoch):
                vs = self.v(s)

                if self.GAE:
                    # use generalized advantage estimation
                    vs_prime = self.v(s_prime).squeeze(dim=-1)
                    assert vs_prime.shape == done_mask.shape
                    vs_target = r + self.gamma * vs_prime * done_mask
                    delta = vs_target - vs.squeeze(dim=-1)
                    delta = delta.detach()
                    advantage_lst = []
                    advantage = 0.0
                    for delta_t in torch.flip(delta, [-1]): # reverse the delta along the time sequence in an episodic data
                        advantage = self.gamma * self.lmbda * advantage + delta_t
                        advantage_lst.append(advantage)
                    advantage_lst.reverse()
                    advantage = torch.tensor(advantage_lst, dtype=torch.float).to(self.device)
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)  # this can have significant improvement (efficiency, stability) on performance

                else:
                    advantage = rewards - vs.squeeze(dim=-1).detach()
                    vs_target = rewards

                logits = self.pi(s)
                if len(logits.shape) > 2:
                    logits = logits.squeeze()
                mean = logits[:, :self.action_dim]
                std = logits[:, self.action_dim:].exp()
                dist = Normal(mean, std)
                dist_entropy = dist.entropy()
                logprob = dist.log_prob(a)
                # pi_a = pi.gather(1,a)
                # ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))
                ratio = torch.exp(logprob.squeeze() - oldlogprob.squeeze())
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage
                loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(vs.squeeze(dim=-1) , vs_target.detach()) - 0.01*dist_entropy
                # loss = -torch.min(surr1, surr2) + 0.5*self.mseLoss(vs.squeeze(dim=-1) , vs_target.detach()) - 0.01*dist_entropy
                
                # print('vs: ', vs.squeeze(dim=-1).shape, vs_target.shape)
                # print('logprob', logprob.shape, ratio.shape, surr1.shape, surr2.shape)
                # print('oldlogprob', oldlogprob.shape)
                # print('advantage: ', advantage.shape)
                # print('loss', loss)

                self.optimizer.zero_grad()
                mean_loss = loss.mean()
                mean_loss.backward()
                self.optimizer.step()

                total_loss += mean_loss.item()

        self.data = [[] for _ in range(self._num_channel)]

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        return total_loss