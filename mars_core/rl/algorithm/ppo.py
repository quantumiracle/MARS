import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from .common.networks import MLP, CNN
from .common.agent import Agent
from .common.rl_utils import choose_optimizer

def PPO(env, args):
    if True: # discrete TODO
        return PPODiscrete(env, args)
    else:
        return None

class PPODiscrete(Agent):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.learning_rate = args.learning_rate
        self.gamma = float(args.algorithm_spec['gamma'])
        self.lmbda = float(args.algorithm_spec['lambda'])
        self.eps_clip = float(args.algorithm_spec['eps_clip'])
        self.K_epoch = args.algorithm_spec['K_epoch']
        self.GAE = args.algorithm_spec['GAE']

        if len(env.observation_space.shape) <= 1:
            self.policy = MLP(env.observation_space, env.action_space, args.net_architecture['policy'], model_for='discrete_policy').to(self.device)
            self.policy_old = copy.deepcopy(self.policy).to(self.device)
            # self.policy_old.load_state_dict(self.policy.state_dict())
            self.value = MLP(env.observation_space, env.action_space, args.net_architecture['value'], model_for='value').to(self.device)

        else:
            self.policy = CNN(env.observation_space, env.action_space, args.net_architecture['policy'], model_for='discrete_policy').to(self.device)
            self.policy_old = copy.deepcopy(self.policy).to(self.device)
            # self.policy_old.load_state_dict(self.policy.state_dict())
            self.value = CNN(env.observation_space, env.action_space, args.net_architecture['value'], model_for='value').to(self.device)

        # cannot use lambda in multiprocessing
        # self.pi = lambda x: self.policy.forward(x, softmax_dim=-1)
        # self.v = lambda x: self.value.forward(x)            

        # TODO a single optimizer for two nets may be problematic
        self.optimizer = choose_optimizer(args.optimizer)(list(self.value.parameters())+list(self.policy.parameters()), lr=float(args.learning_rate))
        self.mseLoss = nn.MSELoss()

        self.num_channel = max(args.num_envs, env.num_agents)
        self.data = [[] for _ in range(self.num_channel)]

    def pi(self, x):
        return self.policy.forward(x)

    def v(self, x):
        return self.value.forward(x)  

    def store(self, transitions):
        # self.data.append(transition)
        # self.data.extend(transitions)
        for i, transition in enumerate(transitions): # iterate over the list
            self.data[i].append(transition)
        
    def make_batch(self, data):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in data:
            s, a, r, s_prime, prob_a, done = transition
            
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append(prob_a)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        # found this step take some time for Pong (not ram), even if no parallel no multiagent
        s,a,r,s_prime,done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float).to(self.device), torch.tensor(a_lst).to(self.device), \
                                          torch.tensor(r_lst).to(self.device), torch.tensor(s_prime_lst, dtype=torch.float).to(self.device), \
                                          torch.tensor(done_lst, dtype=torch.float).to(self.device), torch.tensor(prob_a_lst).to(self.device)
        return s, a, r, s_prime, done_mask, prob_a
        
    def update(self):
        total_loss = 0.
        self.data = [x for x in self.data if x]  # remove empty
        for data in self.data: # iterate over the list of envs
            s, a, r, s_prime, done_mask, oldlogprob = self.make_batch(data)

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
                    vs_target = r + self.gamma * self.v(s_prime) * done_mask
                    delta = vs_target - self.v(s)
                    delta = delta.detach()

                    advantage_lst = []
                    advantage = 0.0
                    for delta_t in torch.flip(delta, [0]):
                        advantage = self.gamma * self.lmbda * advantage + delta_t[0]
                        advantage_lst.append(advantage)
                    advantage_lst.reverse()
                    advantage = torch.tensor(advantage_lst, dtype=torch.float).to(self.device)

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
                # loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(vs , vs_target.detach()) - 0.01*dist_entropy
                loss = -torch.min(surr1, surr2) + 0.5*self.mseLoss(vs.squeeze(dim=-1) , vs_target.detach()) - 0.01*dist_entropy

                self.optimizer.zero_grad()
                mean_loss = loss.mean()
                mean_loss.backward()
                self.optimizer.step()

            total_loss += mean_loss.item()

        self.data = [[] for _ in range(self.num_channel)]

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        return total_loss

    def choose_action(self, s, Greedy=False):
        prob = self.policy(torch.from_numpy(s).unsqueeze(0).float().to(self.device)).squeeze()  # make sure input state shape is correct
        if Greedy:
            a = torch.argmax(prob, dim=-1).item()
            return a
        else:
            dist = Categorical(prob)
            a = dist.sample()
            logprob = dist.log_prob(a)
            return a.detach().item(), logprob.detach().item()

    def save_model(self, path=None):
        torch.save(self.policy.state_dict(), path+'_policy')
        torch.save(self.value.state_dict(), path+'_value')


    def load_model(self, path=None):
        self.policy.load_state_dict(torch.load(path+'_policy'))
        self.policy_old.load_state_dict(self.policy.state_dict())  # important

        self.value.load_state_dict(torch.load(path+'_value'))


class MultiPPODiscrete(nn.Module):
    """ 
    Multi-agent PPO for a single discrete environment.
    Input: observation (dict) {agent_name: agent_observation}
    Output: action (dict) {agent_name: agent_action}
    """
    def __init__(self, agents, observation_spaces, action_spaces, func_approx = 'MLP', fixed_agents=[], learner_args={}, **kwargs):
        super(MultiPPODiscrete, self).__init__()
        self.fixed_agents = fixed_agents
        self.agents = {}
        for agent_name, observation_space, action_space in zip(agents, observation_spaces.values(), action_spaces.values()):
            self.agents[agent_name] = PPODiscrete(observation_space, action_space, func_approx, learner_args, **kwargs).to(learner_args['device'])
        # validation check 
        for agent in fixed_agents:
            assert agent in self.agents

    def store(self, transition):
        (observations, actions, rewards, observations_, logprobs, dones) = transition
        data = (observations.values(), actions.values(), rewards.values(), observations_.values(), logprobs.values(), dones.values())
        for agent_name, *sample in zip(self.agents, *data):
            if agent_name not in self.fixed_agents:
                self.agents[agent_name].store(tuple(sample))
        
    def make_batch(self):
        for agent_name in self.agents:
            if agent_name not in self.fixed_agents:
                self.agents[agent_name].make_batch()

    def update(self):
        for agent_name in self.agents:
            if agent_name not in self.fixed_agents:
                self.agents[agent_name].update()

    def choose_action(self, observations, Greedy=False):
        actions={}
        logprobs={}
        for agent_name in self.agents:
            actions[agent_name], logprobs[agent_name] = self.agents[agent_name].choose_action(observations[agent_name], Greedy)
        return actions, logprobs

    def save_model(self, path=None):
        for idx, agent_name in enumerate(self.agents):
            if agent_name not in self.fixed_agents:
                self.agents[agent_name].save_model(path+'_{}'.format(idx))

    def load_model(self, agent_name=None, path=None):
        if agent_name is not None:  # load model for specific agent only
            self.agents[agent_name].load_model(path)
        else:
            for idx, agent_name in enumerate(self.agents):
                self.agents[agent_name].load_model(path+'_{}'.format(idx))


class ParallelPPODiscrete(nn.Module):
    """
    PPO handles parallel envs wrapped with ***VectorEnv wrapper.
    """
    def __init__(self, num_envs, observation_space, action_space, func_approx = 'MLP', learner_args={}, **kwargs):
        super(ParallelPPODiscrete, self).__init__()
        self.learning_rate = kwargs['learning_rate']
        self.gamma = kwargs['gamma']
        self.lmbda = kwargs['lambda']
        self.eps_clip = kwargs['eps_clip']
        self.K_epoch = kwargs['K_epoch']
        self.device = torch.device(learner_args['device'])
        hidden_dim = kwargs['hidden_dim']
        self.num_envs = num_envs

        self.data = [[] for _ in range(self.num_envs)]
        if func_approx == 'MLP':
            self.policy = PolicyMLP(observation_space, action_space, hidden_dim, self.device).to(self.device)
            self.policy_old = PolicyMLP(observation_space, action_space, hidden_dim, self.device).to(self.device)
            self.policy_old.load_state_dict(self.policy.state_dict())

            self.value = ValueMLP(observation_space, hidden_dim).to(self.device)

        elif func_approx == 'CNN':
            self.policy = PolicyCNN(observation_space, action_space, hidden_dim, self.device).to(self.device)
            self.policy_old = PolicyCNN(observation_space, action_space, hidden_dim, self.device).to(self.device)
            self.policy_old.load_state_dict(self.policy.state_dict())

            self.value = ValueCNN(observation_space, hidden_dim).to(self.device)
        else:
            raise NotImplementedError

        # cannot use lambda in multiprocessing
        # self.pi = lambda x: self.policy.forward(x, softmax_dim=-1)
        # self.v = lambda x: self.value.forward(x)            

        # TODO a single optimizer for two nets may be problematic
        self.optimizer = optim.Adam(list(self.value.parameters())+list(self.policy.parameters()), lr=self.learning_rate, betas=(0.9, 0.999))
        self.mseLoss = nn.MSELoss()

    def pi(self, x):
        return self.policy.forward(x, softmax_dim=-1)

    def v(self, x):
        return self.value.forward(x)  

    def store(self, transition):
        for i, trans in enumerate(transition): # iterate over the list of envs
            self.data[i].append(trans)

    def make_batch(self, env_data):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in env_data:
            s, a, r, s_prime, prob_a, done = transition
            
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append(prob_a)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        # TODO this step take some time for Pong (no ram), better speed it up
        s,a,r,s_prime,done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float).to(self.device), torch.tensor(a_lst).to(self.device), \
                                        torch.tensor(r_lst).to(self.device), torch.tensor(s_prime_lst, dtype=torch.float).to(self.device), \
                                        torch.tensor(done_lst, dtype=torch.float).to(self.device), torch.tensor(prob_a_lst).to(self.device)
        return s,a,r,s_prime,done_mask, prob_a
        
    def train_net(self, GAE=False):
        for data in self.data: # iterate over the list of envs
            s, a, r, s_prime, done_mask, oldlogprob = self.make_batch(data)

            if not GAE:
                rewards = []
                discounted_r = 0
                for reward, is_continue in zip(reversed(r), reversed(done_mask)):
                    if not is_continue:
                        discounted_r = 0
                    discounted_r = reward + self.gamma * discounted_r
                    rewards.insert(0, discounted_r)  # insert in front, cannot use append

                rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            for i in range(self.K_epoch):
                vs = self.v(s)

                if GAE:
                    # use generalized advantage estimation
                    vs_target = r + self.gamma * self.v(s_prime) * done_mask
                    delta = vs_target - self.v(s)
                    delta = delta.detach()

                    advantage_lst = []
                    advantage = 0.0
                    for delta_t in torch.flip(delta, [0]):
                        advantage = self.gamma * self.lmbda * advantage + delta_t[0]
                        advantage_lst.append(advantage)
                    advantage_lst.reverse()
                    advantage = torch.tensor(advantage_lst, dtype=torch.float).to(self.device)

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
                # loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(vs , vs_target.detach()) - 0.01*dist_entropy
                loss = -torch.min(surr1, surr2) + 0.5*self.mseLoss(vs.squeeze(dim=-1) , vs_target.detach()) - 0.01*dist_entropy

                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

        self.data = [[] for _ in range(self.num_envs)]
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

    def choose_action(self, s, Greedy=False):
        prob = self.policy(torch.from_numpy(s).float().to(self.device)).squeeze()  # make sure input state shape is correct
        if Greedy:
            a = torch.argmax(prob, dim=-1).item()
            return a
        else:
            dist = Categorical(prob)
            a = dist.sample()
            logprob = dist.log_prob(a)
            return a.detach().cpu().numpy(), logprob.detach().cpu().numpy()

    def save_model(self, path=None):
        torch.save(self.policy.state_dict(), path+'_policy')
        torch.save(self.value.state_dict(), path+'_value')


    def load_model(self, path=None):
        self.policy.load_state_dict(torch.load(path+'_policy'))
        self.policy_old.load_state_dict(self.policy.state_dict())  # important

        self.value.load_state_dict(torch.load(path+'_value'))

class ParallelMultiPPODiscrete(nn.Module):
    """
    Multi-agent PPO for multiple parallel discrete environments (wrapped with ***VectorEnv wrapper).
    Input: observation (list of dict) [{agent_name: agent_observation}, {agent_name: agent_observation}], each dict for a single env;
    Output: action (list of dict) [{agent_name: agent_action}, {agent_name: agent_observation}], each dict for a single env.
    """
    def __init__(self, num_envs, agents, observation_spaces, action_spaces, func_approx = 'MLP', fixed_agents=[], learner_args={}, **kwargs):
        super(ParallelMultiPPODiscrete, self).__init__()
        self.fixed_agents = fixed_agents
        self.num_envs = num_envs
        self.agents = {}
        for agent_name, observation_space, action_space in zip(agents, observation_spaces.values(), action_spaces.values()):
            self.agents[agent_name] = ParallelPPODiscrete(num_envs, observation_space, action_space, func_approx, learner_args, **kwargs).to(learner_args['device'])
        # validation check 
        for agent in fixed_agents:
            assert agent in self.agents

    def store(self, transition):
        (observations, actions, rewards, observations_, logprobs, dones) = transition
        
        data = [[] for _ in range(len(self.agents))]  # first dimension is the agent
        for obs, act, r, obs_, logp, d in zip(observations, actions, rewards, observations_, logprobs, dones): # iterate over envs
            env_data = (obs.values(), act.values(), r.values(), obs_.values(), logp.values(), d.values())
            for j, *agent_data in enumerate(zip(*env_data)): # iterate over agents
                data[j].append(*agent_data)  # *agent_data is a tuple

        for agent_name, sample in zip(self.agents, data): # each sample is a list of data (containing different envs) for one agent
            if agent_name not in self.fixed_agents:
                self.agents[agent_name].store(sample)
        
    def make_batch(self):
        for agent_name in self.agents:
            if agent_name not in self.fixed_agents:
                self.agents[agent_name].make_batch()

    def train_net(self, GAE=False):
        for agent_name in self.agents:
            if agent_name not in self.fixed_agents:
                # print('trained agents: ', agent_name)
                self.agents[agent_name].train_net(GAE)

    def choose_action(self, observations, Greedy=False):
        multi_actions = []
        multi_logprobs = []
        empty_dict_in_obs = [bool(x) is False for x in observations]  # for example, if returns [False, True], then second env provides empty observation {}
        observations=list(filter(None, observations)) # filter out empty dict in observations

        # looping through both the envs and agents is slow
        # for obs in observations: # iterate over the list of envs
        #     actions={}
        #     logprobs={}
        #     for agent_name in self.agents:
        #         actions[agent_name], logprobs[agent_name] = self.agents[agent_name].choose_action(obs[agent_name], Greedy)
        #     multi_actions.append(actions)
        #     multi_logprobs.append(logprobs)
        # return multi_actions, multi_logprobs

        # concatenate the obs for different envs by the same agent and take one forward inference as a whole for each agent
        obs = np.array([list(obs.values()) for obs in observations]).swapaxes(0,1) # shape after swap: (# agents, # envs, obs_dim)
        for i, agent_name in enumerate(self.agents):
            actions, logprobs = self.agents[agent_name].choose_action(obs[i], Greedy)  # one forward inference for multiple envs
            if len(actions.shape)<1:
                multi_actions.append(actions.reshape(-1))  # expand len(shape) from 0 to 1 for single env case
                multi_logprobs.append(logprobs.reshape(-1))
            else:
                multi_actions.append(actions.reshape(-1)) 
                multi_logprobs.append(logprobs) # multi_actions shape: (# agents, # envs)

        actions_list, logprobs_list = [{} for _ in range(self.num_envs)], [{} for _ in range(self.num_envs)]
        for i, agent_name in enumerate(self.agents):
            available_action_idx = 0
            for j in range(self.num_envs):
                if empty_dict_in_obs[j]: # the j-th env provides empty dict as observation, then returns empty dict as action and log_probs
                    actions_list[j][agent_name] = {}
                    logprobs_list[j][agent_name] = {}
                else:
                    actions_list[j][agent_name] = multi_actions[i][available_action_idx]
                    logprobs_list[j][agent_name] = multi_logprobs[i][available_action_idx]
                    available_action_idx += 1

        return actions_list, logprobs_list

    def save_model(self, path=None):
        for idx, agent_name in enumerate(self.agents):
            if agent_name not in self.fixed_agents:
                
                self.agents[agent_name].save_model(path+'_{}'.format(idx))

    def load_model(self, agent_name=None, path=None):
        if agent_name is not None:  # load model for specific agent only
            self.agents[agent_name].load_model(path)
        else:
            for idx, agent_name in enumerate(self.agents):
                self.agents[agent_name].load_model(path+'_{}'.format(idx))



