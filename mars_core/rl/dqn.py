import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random, copy
from .agent import Agent
from .nn_components import cReLU, Flatten
from .storage import ReplayBuffer
from .rl_utils import choose_optimizer, EpsilonScheduler
from .networks import NetBase
class DQN(Agent):
    """
    DQN algorithm
    """
    def __init__(self, env, args):
        super().__init__(env, args)
        self.model = self._select_type(env, args).to(self.device)
        self.target = copy.deepcopy(self.model).to(self.device)

        self.buffer = ReplayBuffer(int(float(args.algorithm_spec['replay_buffer_size']))) # first float then int to handle the scientific number like 1e5
        self.optimizer = choose_optimizer(args.optimizer)(self.model.parameters(), lr=float(args.learning_rate))
        self.epsilon_scheduler = EpsilonScheduler(args.algorithm_spec['eps_start'], args.algorithm_spec['eps_final'], args.algorithm_spec['eps_decay'])
        self.schedulers = [self.epsilon_scheduler]

        self.gamma = float(args.algorithm_spec['gamma'])
        self.multi_step = args.algorithm_spec['multi_step']  # TODO
        self.target_update_interval = args.algorithm_spec['target_update_interval']

        self.update_cnt = 1

    def _select_type(self, env, args):
        if args.num_envs == 1:
            if args.algorithm_spec['dueling']:
                model = DuelingDQN(env, args.hidden_dim)
            else:
                model = DQNBase(env, args.hidden_dim)
        else:
            if args.algorithm_spec['dueling']:
                model = ParallelDuelingDQN(env, args.hidden_dim, args.num_envs)
            else:
                model = ParallelDQN(env, args.hidden_dim, args.num_envs)
        return model

    def choose_action(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon_scheduler.get_epsilon()
        if not isinstance(state, torch.Tensor):
            state = torch.Tensor(state).to(self.device)
        action = self.model.choose_action(state, epsilon)
        return action

    def store(self, sample):
        self.buffer.push(*sample)

    @property
    def ready_to_update(self):
        if len(self.buffer) > self.batch_size:
            return True
        else:
            return False

    def update(self):
        state, action, reward, next_state, done = self.buffer.sample(self.batch_size)
        weights = torch.ones(self.batch_size)

        state = torch.FloatTensor(np.float32(state)).to(self.device)
        next_state = torch.FloatTensor(np.float32(next_state)).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(done).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        # Q-Learning with target network
        q_values = self.model(state)
        target_next_q_values = self.target(next_state)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = target_next_q_values.max(1)[0]
        expected_q_value = reward + (self.gamma ** self.multi_step) * next_q_value * (1 - done)

        # Huber Loss
        loss = F.smooth_l1_loss(q_value, expected_q_value.detach(), reduction='none')
        loss = (loss * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.update_cnt % self.target_update_interval == 0:
            self.update_target(self.model, self.target)
            self.update_cnt = 0
        self.update_cnt += 1

        return loss.item()

    def save_model(self, path):
        torch.save(self.policy.state_dict(), path+'_model')
        torch.save(self.policy.state_dict(), path+'_target')

    def load_model(self, path, eval=True):
        self.model.load_state_dict(torch.load(path+'_model'))
        self.target.load_state_dict(torch.load(path+'_target'))

        if eval:
            self.model.eval()
            self.target.eval()


class DQNBase(NetBase):
    """
    Basic Q network

    parameters
    ---------
    env         environment(openai gym)
    """
    def __init__(self, env, hidden_dim=64):
        super().__init__(env)
        self.construct_net(hidden_dim, nn.Tanh())

    def construct_net(self, hidden_dim, activation=nn.ReLU()):
        self.flatten = Flatten()
        self.hidden_dim = hidden_dim
        
        if len(self._observation_shape) <= 1: # not image
            self.features = nn.Sequential(
                nn.Linear(self._observation_shape[0], hidden_dim),
                activation,
                nn.Linear(hidden_dim, hidden_dim),
                activation,
                nn.Linear(hidden_dim, hidden_dim),
                activation,
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(self._observation_shape[0], 8, kernel_size=4, stride=2),
                activation,
                nn.Conv2d(16, 8, kernel_size=5, stride=1),
                activation,
                nn.Conv2d(16, 8, kernel_size=3, stride=1),
                activation,
            )
        
        self.fc = nn.Sequential(
            nn.Linear(self._feature_size(), hidden_dim),
            activation,
            nn.Linear(hidden_dim, self._action_shape)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
    
    def _feature_size(self):
        if isinstance(self._observation_shape, int):
            return self.features(torch.zeros(1, self._observation_shape)).view(1, -1).size(1)
        else:
            return self.features(torch.zeros(1, *self._observation_shape)).view(1, -1).size(1)
    
    def choose_action(self, state, epsilon=0.):
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
            action = random.randrange(self._action_shape)
        return action

class DuelingDQN(DQNBase):
    """
    Dueling Network Architectures for Deep Reinforcement Learning
    https://arxiv.org/abs/1511.06581
    """
    def __init__(self, env, hidden_dim=64, activation=nn.Tanh(), **kw):
        super().__init__(env, hidden_dim, **kw)
        self.advantage = self.fc

        self.value = nn.Sequential(
            nn.Linear(self._feature_size(), hidden_dim),
            activation,
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
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

    def choose_action(self, state, epsilon):
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
            action = np.random.randint(self._action_shape, size=self.number_envs)
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

# class ParallelNashDQN(DQNBase):
#     """
#     Nash-DQN for parallel env sampling

#     parameters
#     ---------
#     env         environment(openai gym)
#     """
#     def __init__(self, env, hidden_dim=64, number_envs=2):
#         super(ParallelNashDQN, self).__init__(env, hidden_dim)
#         self.number_envs = number_envs
#         try:
#             self.input_shape = tuple(map(operator.add, env.observation_space.shape, env.observation_space.shape)) # double the shape
#             self.num_actions = (env.action_space.n)**2
#         except:
#             self.input_shape = tuple(map(operator.add, env.observation_space[0].shape, env.observation_space[0].shape)) # double the shape
#             self.num_actions = (env.action_space[0].n)**2
#         self.construct_net(hidden_dim)

#     def act(self, state, epsilon, q_value=True):
#         """
#         Parameters
#         ----------
#         state       torch.Tensor with appropritate device type
#         epsilon     epsilon for epsilon-greedy
#         """
#         if random.random() > epsilon:  # NoisyNet does not use e-greedy
#             with torch.no_grad():
#                 q_value = self.forward(state)
#                 action = q_value.max(1)[1].detach().cpu().numpy()
#         else:
#             action = np.random.randint(self.num_actions, size=self.number_envs)
#         if q_value:
#             return action, q_value
#         else:
#             return action    

# class Policy(DQNBase):
#     """
#     Policy with only actors. This is used in supervised learning for NFSP.
#     """
#     def __init__(self, env, hidden_dim=64):
#         super(Policy, self).__init__(env, hidden_dim)
#         self.fc = nn.Sequential(
#             nn.Linear(self._feature_size(), int(hidden_dim/2)),
#             nn.ReLU(),
#             nn.Linear(int(hidden_dim/2), self.num_actions),
#             nn.Softmax(dim=1)
#         )

#     def act(self, state):
#         """
#         Parameters
#         ----------
#         state       torch.Tensor with appropritate device type
#         """
#         with torch.no_grad():
#             state = state.unsqueeze(0)
#             distribution = self.forward(state)
#             action = distribution.multinomial(1).item()
#         return action

