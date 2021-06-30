import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random
import math
import operator
from functools import partial

def DQN(env, args, Nash=False):
    if Nash:
        if args.num_envs == 1:
            # pass # TODO
            model = ParallelNashDQN(env, args.hidden_dim, args.num_envs)
        else:
            model = ParallelNashDQN(env, args.hidden_dim, args.num_envs)

    else:
        if args.num_envs == 1:
            if args.dueling:
                model = DuelingDQN(env, args.hidden_dim)
            else:
                model = DQNBase(env, args.hidden_dim)
        else:
            if args.dueling:
                model = ParallelDuelingDQN(env, args.hidden_dim, args.num_envs)
            else:
                model = ParallelDQN(env, args.hidden_dim, args.num_envs)
    return model

class DQNBase(nn.Module):
    """
    Basic DQN

    parameters
    ---------
    env         environment(openai gym)
    """
    def __init__(self, env, hidden_dim=64):
        super(DQNBase, self).__init__()
        try:
            self.input_shape = env.observation_space.shape
            self.num_actions = env.action_space.n
        except:
            self.input_shape = env.observation_space[0].shape
            self.num_actions = env.action_space[0].n
        print(self.input_shape)
        self.construct_net(hidden_dim, nn.Tanh())

    def construct_net(self, hidden_dim, activation=nn.ReLU()):
        self.flatten = Flatten()
        self.hidden_dim = hidden_dim
        # if isinstance(self.input_shape, int):  # discrete observation
        #     self.features = nn.Sequential(
        #         nn.Linear(self.input_shape, hidden_dim),
        #         activation,
        #         nn.Linear(hidden_dim, hidden_dim),
        #         activation,
        #         # nn.Linear(hidden_dim, int(hidden_dim/2)),
        #         # nn.ReLU(),
        #     )
        # elif not len(self.input_shape) > 1: # not image
        
        if not len(self.input_shape) > 1: # not image
            self.features = nn.Sequential(
                nn.Linear(self.input_shape[0], hidden_dim),
                activation,
                nn.Linear(hidden_dim, hidden_dim),
                activation,
                # nn.Linear(hidden_dim, int(hidden_dim/2)),
                # nn.ReLU(),
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(self.input_shape[0], 8, kernel_size=4, stride=2),
                cReLU(),
                nn.Conv2d(16, 8, kernel_size=5, stride=1),
                cReLU(),
                nn.Conv2d(16, 8, kernel_size=3, stride=1),
                cReLU()
            )
        
        self.fc = nn.Sequential(
            nn.Linear(self._feature_size(), int(hidden_dim/2)),
            activation,
            nn.Linear(int(hidden_dim/2), self.num_actions)
        )
        
    def forward(self, x):
        if len(x.shape)>2:  # image input: len(x.shape) == 4
            x /= 255.
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
    
    def _feature_size(self):
        # print(torch.zeros(1, *self.input_shape).shape)
        if isinstance(self.input_shape, int):
            return self.features(torch.zeros(1, self.input_shape)).view(1, -1).size(1)
        else:
            return self.features(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)
    
    def act(self, state, epsilon):
        """
        Parameters
        ----------
        state       torch.Tensor with appropritate device type
        epsilon     epsilon for epsilon-greedy
        """
        if random.random() > epsilon:  # NoisyNet does not use e-greedy
            with torch.no_grad():
                state   = state.unsqueeze(0)
                q_value = self.forward(state)
                action  = q_value.max(1)[1].item()
        else:
            action = random.randrange(self.num_actions)
        return action

class DuelingDQN(DQNBase):
    """
    Dueling Network Architectures for Deep Reinforcement Learning
    https://arxiv.org/abs/1511.06581
    """
    def __init__(self, env, hidden_dim=64, activation=nn.Tanh(), **kw):
        super(DuelingDQN, self).__init__(env, hidden_dim, **kw)
        self.advantage = self.fc

        self.value = nn.Sequential(
            nn.Linear(self._feature_size(), int(hidden_dim/2)),
            activation,
            nn.Linear(int(hidden_dim/2), 1)
        )
    
    def forward(self, x):
        if len(x.shape)>2:  # image input: len(x.shape) == 4
            x /= 255.
        x = self.features(x)
        x = self.flatten(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean(1, keepdim=True)

class ParallelDQN(DQNBase):
    """
    DQN for parallel env sampling

    parameters
    ---------
    env         environment(openai gym)
    """
    def __init__(self, env, hidden_dim=64, number_envs=2):
        super(ParallelDQN, self).__init__(env, hidden_dim)
        self.number_envs = number_envs

    def act(self, state, epsilon):
        """
        Parameters
        ----------
        state       torch.Tensor with appropritate device type
        epsilon     epsilon for epsilon-greedy
        """
        if random.random() > epsilon:  # NoisyNet does not use e-greedy
            with torch.no_grad():
                q_value = self.forward(state)
                action = q_value.max(1)[1].detach().cpu().numpy()
        else:
            action = np.random.randint(self.num_actions, size=self.number_envs)
        return action

class ParallelNashDQN(DQNBase):
    """
    Nash-DQN for parallel env sampling

    parameters
    ---------
    env         environment(openai gym)
    """
    def __init__(self, env, hidden_dim=64, number_envs=2):
        super(ParallelNashDQN, self).__init__(env, hidden_dim)
        self.number_envs = number_envs
        try:
            self.input_shape = tuple(map(operator.add, env.observation_space.shape, env.observation_space.shape)) # double the shape
            self.num_actions = (env.action_space.n)**2
        except:
            self.input_shape = tuple(map(operator.add, env.observation_space[0].shape, env.observation_space[0].shape)) # double the shape
            self.num_actions = (env.action_space[0].n)**2
        self.construct_net(hidden_dim)

    def act(self, state, epsilon, q_value=True):
        """
        Parameters
        ----------
        state       torch.Tensor with appropritate device type
        epsilon     epsilon for epsilon-greedy
        """
        if random.random() > epsilon:  # NoisyNet does not use e-greedy
            with torch.no_grad():
                q_value = self.forward(state)
                action = q_value.max(1)[1].detach().cpu().numpy()
        else:
            action = np.random.randint(self.num_actions, size=self.number_envs)
        if q_value:
            return action, q_value
        else:
            return action   

class ParallelDuelingDQN(DuelingDQN, ParallelDQN):
    """
    DuelingDQN for parallel env sampling

    parameters
    ---------
    env         environment(openai gym)

    Note: for mulitple inheritance, see a minimal example:

    class D:
        def __init__(self,):
            super(D, self).__init__()
            self.a=1
        def f(self):
            pass
                
        def f1(self):
            pass

    class A(D):
        def __init__(self,):
            super(A, self).__init__()
            self.a=1
            
        def f1(self):
            self.a+=2
            print(self.a)
        
    class B(D):
        def __init__(self,):
            super(B, self).__init__()
            self.a=1
        def f(self):
            self.a-=1
            print(self.a)
        
    class C(B,A):
        def __init__(self,):
            super(C, self).__init__()   
            

    c=C()
    c.f1() 

    => 3
    """
    def __init__(self, env, hidden_dim=64, number_envs=2):
        super(ParallelDuelingDQN, self).__init__(env=env, hidden_dim=hidden_dim, number_envs=number_envs)
     

class Policy(DQNBase):
    """
    Policy with only actors. This is used in supervised learning for NFSP.
    """
    def __init__(self, env, hidden_dim=64):
        super(Policy, self).__init__(env, hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(self._feature_size(), int(hidden_dim/2)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim/2), self.num_actions),
            nn.Softmax(dim=1)
        )

    def act(self, state):
        """
        Parameters
        ----------
        state       torch.Tensor with appropritate device type
        """
        with torch.no_grad():
            state = state.unsqueeze(0)
            distribution = self.forward(state)
            action = distribution.multinomial(1).item()
        return action


class cReLU(nn.Module):
    def forward(self, x):
        return torch.cat((F.relu(x), F.relu(-x)), dim=1)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

