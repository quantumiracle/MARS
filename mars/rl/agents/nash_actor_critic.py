import copy
from math import log
import gym
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torch.distributions import Categorical, Normal, MultivariateNormal
import numpy as np
from .agent import Agent
from ..common.networks import MLP, CNN, get_model
from ..common.rl_utils import choose_optimizer, EpsilonScheduler
from ..common.storage import ReplayBuffer
from mars.utils.typing import List, Tuple, StateType, ActionType, SampleType, SingleEnvMultiAgentSampleType

torch.autograd.set_detect_anomaly(True)

class NashActorCritic(Agent):
    """ 
    Nash Actor Critic algorithm
    """    
    def __init__(self, env, args):
        super().__init__(env, args)
        self.gamma = float(args.algorithm_spec['gamma'])
        self.multi_step = args.algorithm_spec['multi_step'] 
        self.target_update_interval = args.algorithm_spec['target_update_interval']
        if args.num_process > 1:
            self.buffer = args.add_components['replay_buffer']
        else:
            self.buffer = ReplayBuffer(int(float(args.algorithm_spec['replay_buffer_size'])), \
                args.algorithm_spec['multi_step'], args.algorithm_spec['gamma'], args.num_envs, args.batch_size) # first float then int to handle the scientific number like 1e5
        self.epsilon_scheduler = EpsilonScheduler(args.algorithm_spec['eps_start'], args.algorithm_spec['eps_final'], args.algorithm_spec['eps_decay'])
        self.schedulers.append(self.epsilon_scheduler)

        self.num_agents = env.num_agents[0] if isinstance(env.num_agents, list) else env.num_agents
        self.env = env
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._init_model(env, args)
        print(f'Actor networks: ', self.actors,
                '\nTarget actor networks: ', self.target_actors,
                '\nCritic network: ', self.critic,
                '\nTarget critic network: ', self.target_critic,
            )
        actor_params = []
        for a in self.actors:
            actor_params += list(a.parameters())
        self.max_player_optimizer = choose_optimizer(args.optimizer)(self.actors[0].parameters(), lr=float(args.learning_rate))
        self.min_player_optimizer = choose_optimizer(args.optimizer)(self.actors[1].parameters(), lr=float(args.learning_rate))
        self.critic_optimizer = choose_optimizer(args.optimizer)(self.critic.parameters(), lr=float(args.learning_rate))
        self.mseLoss = nn.MSELoss()
        self.log_std_min = -20
        self.log_std_max = 2
        self.update_cnt = 1

    @property
    def ready_to_update(self):
        return True if self.buffer.get_len() > self.batch_size else False

    def store(self, sample: SampleType) -> None:
        """ Store samples in buffer.

        :param sample: a list of samples from different environments (if using parallel env)
        :type sample: SampleType
        """ 
        self.buffer.push(sample)


    def _init_model(self, env, args):
        self.actors = []
        self.target_actors = []
        if len(self.observation_space.shape) <= 1:
            # self.feature_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape = (args.net_architecture['feature']['hidden_dim_list'][-1],))
            self.sa_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape = (self.observation_dim+2*self.action_dim,))
        else:
            self.feature_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape = (256,))
            self.sa_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape = (256+2*self.action_dim,))
            self.feature = CNN(self.observation_space, self.feature_space, args.net_architecture['feature'], model_for='feature').to(self.device)
        
        for _ in range(env.num_agents): # two actors share the feature with critic
            actor = MLP(self.observation_space, env.action_space, args.net_architecture['actor'], model_for='gaussian_policy').to(self.device)
            self.actors.append(actor)            
            self.target_actors.append(copy.deepcopy(actor).to(self.device))

        self.critic = MLP(self.sa_space, env.action_space, args.net_architecture['critic'], model_for='continuous_q').to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)

        if args.num_process > 1:
            for i in self.num_agents:
                self.actors[i].share_memory()
                self.target_actors[i].share_memory()
            self.critic.share_memory()
            self.target_critic.share_memory()

    def choose_action(self, state, by_target=False, track_grad=False, Greedy=False, epsilon=None):
        actions = []
        actors = self.target_actors if by_target else self.actors

        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        if Greedy:
            for actor in actors:
                logits = actor(state)

                if len(logits.shape) > 2:
                    logits = logits.squeeze()
                mean = torch.tanh(logits[:, :self.action_dim])    
                log_std = logits[:, self.action_dim:]  # no tanh on log var
                log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)  # clipped to prevent blowing std
                std = log_std.exp()    
                normal = Normal(mean, std)
                a = normal.sample()    
                actions.append(a)

            if track_grad:
                actions = torch.stack(actions)  # (#agents, batch_size, action_dim)
            else:
                actions = [a.detach().cpu().numpy() for a in actions] 

        else:
            if epsilon is None:
                epsilon = self.epsilon_scheduler.get_epsilon()
            if random.random() > epsilon:  # NoisyNet does not use e-greedy     
                if self.num_env > 1:
                    actions = [[self.action_space.sample() for _ in range(len(self.env.action_space))] for _ in range(self.num_agents)] # (#agents, env, action_dim)
                else:
                    actions = [self.action_space.sample() for _ in range(self.num_agents)]
            else:
                for actor in actors:
                    logits = actor(state)
                    if len(logits.shape) > 2:
                        logits = logits.squeeze()
                    mean = torch.tanh(logits[:, :self.action_dim])
                    log_std = logits[:, self.action_dim:]  # no tanh on log var
                    log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)  # clipped to prevent blowing std
                    std = log_std.exp()
                    normal = Normal(mean, std)
                    a = normal.sample()
                    actions.append(a)

                if track_grad:
                    actions = torch.stack(actions)  # (#agents, batch_size, action_dim)
                else:
                    actions = [a.detach().cpu().numpy() for a in actions] 
                        
        return actions # (agents, envs, action_dim)


    def update(self):
        infos = {}
        state, action, reward, next_state, done = self.buffer.sample(self.batch_size)

        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.IntTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(np.float32(done)).to(self.device)

        # update the critic
        sa = torch.cat([state, action], dim=-1)
        q_value = self.critic(sa).squeeze()
        with torch.no_grad():
            next_action = self.choose_action(next_state, by_target=True, Greedy=True, track_grad=True)
            next_action = next_action.swapaxes(0,1).reshape(next_state.shape[0], -1) # to (batch_size, #agents*action_dim)
            s_a_ = torch.cat([next_state, next_action], dim=-1)
            next_q_value = self.target_critic(s_a_).squeeze()
            expected_q_value = reward + (self.gamma ** self.multi_step) * next_q_value * (1 - done)
        critic_loss = F.mse_loss(q_value, expected_q_value.detach(), reduction='none').mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update the actors
        new_action = self.choose_action(state, by_target=True, Greedy=True, track_grad=True)
        new_action = new_action.swapaxes(0,1).reshape(state.shape[0], -1)
        sa_ = torch.cat([state, new_action], dim=-1)
        new_q_value = self.critic(sa_).mean()
        max_player_loss = - new_q_value
        min_player_loss = new_q_value

        self.max_player_optimizer.zero_grad()
        max_player_loss.backward(retain_graph=True)  # critic graph needs to be used twice
        self.max_player_optimizer.step()

        self.min_player_optimizer.zero_grad()
        min_player_loss.backward()
        self.min_player_optimizer.step()

        if self.update_cnt % self.target_update_interval == 0:
            self.update_target(self.critic, self.target_critic)
            for i in range(self.num_agents):
                self.update_target(self.actors[i], self.target_actors[i])
        self.update_cnt += 1

        loss = critic_loss.item()

        infos[f'critic loss'] = critic_loss
        infos[f'actor loss max player'] = max_player_loss
        infos[f'actor loss max player'] = min_player_loss

        return loss, infos

    def save_model(self, path):
        torch.save(self.critic.state_dict(), path+'_critic')
        torch.save(self.target_critic.state_dict(), path+'_target_critic')
        for i in range(self.num_agents):
            torch.save(self.actors[i].state_dict(), path+f'_actor{i}')
            torch.save(self.target_actors[i].state_dict(), path+f'_target_actor{i}')

    def load_model(self, path):
        self.critic.load_state_dict(torch.load(path+'_critic'))
        self.target_critic.load_state_dict(torch.load(path+'_target_critic'))
        for i in range(self.num_agents):
            self.actors[i].load_state_dict(torch.load(path+f'_actor{i}'))
            self.target_actors[i].load_state_dict(torch.load(path+f'_target_actor{i}'))
            self.actors[i].eval()
            self.target_actors[i].eval()
        self.critic.eval()
        self.target_critic.eval()
