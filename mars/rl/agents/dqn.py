import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random, copy
from .agent import Agent
from ..common.storage import ReplayBuffer
from ..common.rl_utils import choose_optimizer, EpsilonScheduler
from ..common.networks import NetBase, get_model
from mars.utils.typing import List, Union, StateType, ActionType, SampleType, SamplesType, ConfigurationDict

class DQN(Agent):
    """
    DQN algorithm
    """
    def __init__(self, env, args):
        super().__init__(env, args)
        self.model = self._select_type(env, args).to(self.device)
        print(self.model)
        self.target = copy.deepcopy(self.model).to(self.device)
        
        if args.num_process > 1:
            self.model.share_memory()
            self.target.share_memory()
            self.buffer = args.add_components['replay_buffer']
        else:
            self.buffer = ReplayBuffer(int(float(args.algorithm_spec['replay_buffer_size']))) # first float then int to handle the scientific number like 1e5

        self.update_target(self.model, self.target)

        self.optimizer = choose_optimizer(args.optimizer)(self.model.parameters(), lr=float(args.learning_rate))
        self.epsilon_scheduler = EpsilonScheduler(args.algorithm_spec['eps_start'], args.algorithm_spec['eps_final'], args.algorithm_spec['eps_decay'])
        self.schedulers.append(self.epsilon_scheduler)

        self.gamma = float(args.algorithm_spec['gamma'])
        self.multi_step = args.algorithm_spec['multi_step']  # TODO
        self.target_update_interval = args.algorithm_spec['target_update_interval']

        self.update_cnt = 1

    def _select_type(self, env, args):
        if args.num_envs == 1:
            if args.algorithm_spec['dueling']:
                model = DuelingDQN(env, args.net_architecture)
            else:
                model = DQNBase(env, args.net_architecture)
        else:
            if args.algorithm_spec['dueling']:
                model = ParallelDuelingDQN(env, args.net_architecture, args.num_envs)
            else:
                model = ParallelDQN(env, args.net_architecture, args.num_envs)
        return model

    def reinit(self, nets_init=False, buffer_init=True, schedulers_init=True):
        if nets_init:
            self.model.reinit()  # reinit the networks seem to hurt the overall learning performance
            self.target.reinit()
            self.update_target(self.model, self.target)
        if buffer_init:
            self.buffer.clear()
        if schedulers_init:
            for scheduler in self.schedulers:
                scheduler.reset()

    def choose_action(
        self, 
        state: List[StateType], 
        Greedy: bool = False, 
        epsilon: Union[float, None] = None
        ) -> List[ActionType]:
        """Choose action give state.

        :param state: observed state from the agent
        :type state: List[StateType]
        :param Greedy: whether adopt greedy policy (no randomness for exploration) or not, defaults to False
        :type Greedy: bool, optional
        :param epsilon: parameter value for \epsilon-greedy, defaults to None
        :type epsilon: Union[float, None], optional
        :return: the actions
        :rtype: List[ActionType]
        """
        if Greedy:
            epsilon = 0.
        elif epsilon is None:
            epsilon = self.epsilon_scheduler.get_epsilon()
        if not isinstance(state, torch.Tensor):
            state = torch.Tensor(state).to(self.device)
        action = self.model.choose_action(state, epsilon)

        return action

    def store(self, sample: SampleType) -> None:
        """ Store samples in buffer.

        :param sample: a list of samples from different environments (if using parallel env)
        :type sample: SampleType
        """ 
        # self.buffer.push(*sample)
        self.buffer.push(sample)

    @property
    def ready_to_update(self):
        # return True if len(self.buffer) > self.batch_size else False
        return True if self.buffer.get_len() > self.batch_size else False

    def update(self):
        state, action, reward, next_state, done = self.buffer.sample(self.batch_size)
        weights = torch.ones(self.batch_size)

        state = torch.FloatTensor(np.float32(state)).to(self.device)
        next_state = torch.FloatTensor(np.float32(next_state)).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(np.float32(done)).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        # reward normalization
        # reward =  (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6)

        # Q-Learning with target network
        q_values = self.model(state)
        target_next_q_values = self.target(next_state)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = target_next_q_values.max(1)[0]

        # additional value normalization (this effectively prevent increasing Q/loss value)
        # next_q_value =  (next_q_value - next_q_value.mean(dim=0)) / (next_q_value.std(dim=0) + 1e-6)
        expected_q_value = reward + (self.gamma ** self.multi_step) * next_q_value * (1 - done)
        # Huber Loss
        loss = F.smooth_l1_loss(q_value, expected_q_value.detach(), reduction='none')  # slimevolley env only works with this!
        # loss = F.mse_loss(q_value, expected_q_value.detach())

        # loss = loss.mean()
        loss = (loss * weights).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.update_cnt % self.target_update_interval == 0:
            self.update_target(self.model, self.target)
            # self.update_cnt = 0
        self.update_cnt += 1

        return loss.detach().item()

    def save_model(self, path):
        try:  # for PyTorch >= 1.7 to be compatible with loading models from any lower version
            torch.save(self.model.state_dict(), path+'_model', _use_new_zipfile_serialization=False) 
            torch.save(self.target.state_dict(), path+'_target', _use_new_zipfile_serialization=False)
        except:  # for lower versions
            torch.save(self.model.state_dict(), path+'_model')
            torch.save(self.target.state_dict(), path+'_target')

    def load_model(self, path, eval=True):
        self.model.load_state_dict(torch.load(path+'_model'))
        self.target.load_state_dict(torch.load(path+'_target'))

        if eval:
            self.model.eval()
            self.target.eval()

class DQNBase(NetBase):
    """Basic Q network

    :param env: env object
    :type env: object
    :param net_args: network architecture arguments
    :type net_args: dict
    """
    def __init__(self, env, net_args):
        super().__init__(env.observation_space, env.action_space)
        self._construct_net(env, net_args)

    def _construct_net(self, env, net_args):
        if len(self._observation_shape) <= 1: # not image
            self.net = get_model('mlp')(env.observation_space, env.action_space, net_args, model_for='discrete_q')
        else:
            self.net = get_model('cnn')(env.observation_space, env.action_space, net_args, model_for='discrete_q')
    
    def reinit(self, ):
        self.net.reinit()

    def forward(self, x):
        return self.net(x)

    def choose_action(self, state, epsilon=0.):
        """Choose action acoording to state.

        :param state: state/observation input
        :type state:  torch.Tensor
        :param epsilon: epsilon for epsilon-greedy, defaults to 0.
        :type epsilon: float, optional
        :return: action
        :rtype: int or np.ndarray
        """        
        if random.random() > epsilon:  # NoisyNet does not use e-greedy
            with torch.no_grad():
                state   = state.unsqueeze(0)
                q_value = self.net(state)
                action  = q_value.max(1)[1].item()
        else:
            action = random.randrange(self._action_shape)
        return action


class DuelingDQN(DQNBase):
    """
    Dueling Network Architectures for Deep Reinforcement Learning
    https://arxiv.org/abs/1511.06581
    """
    def __init__(self, env, net_args, **kwargs):
        super().__init__(env, net_args, **kwargs)
        self._construct_net(env, net_args)
    
    def _construct_net(self, env, net_args):
        # Here I use separate networks for advantage and value heads 
        # due to the usage of internal network builder, they should use
        # a shared network body with two heads.
        if len(self._observation_shape) <= 1: # not image
            self.advantage = get_model('mlp')(env.observation_space, env.action_space, net_args, model_for='discrete_q')
            self.value = get_model('mlp')(env.observation_space, env.action_space, net_args, model_for='discrete_q')
        else:  
            self.advantage = get_model('cnn')(env.observation_space, env.action_space, net_args, model_for='discrete_q')
            self.value = get_model('cnn')(env.observation_space, env.action_space, net_args, model_for='discrete_q')

    def reinit(self, ):
        self.advantage.reinit()
        self.value.reinit()

    def net(self, x):
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean(1, keepdim=True)

# class DuelingDQN(DQNBase):
#     """
#     Dueling Network Architectures for Deep Reinforcement Learning
#     https://arxiv.org/abs/1511.06581
#     Different from above, a self-contained version, using a shared network body for advantage and value heads.
#     """
#     def __init__(self, env, hidden_dim=64, activation=nn.Tanh(), **kwargs):
#         super().__init__(env, hidden_dim, **kwargs)
#         self.construct_net(hidden_dim, nn.Tanh())
        
#         self.advantage = self.fc

#         self.value = nn.Sequential(
#             nn.Linear(self._feature_size(), hidden_dim),
#             activation,
#             nn.Linear(hidden_dim, 1))
        
#     def construct_net(self, hidden_dim=64, activation=nn.ReLU()):
#         self.flatten = Flatten()
#         activation = nn.Tanh()
        
#         if len(self._observation_shape) <= 1: # not image
#             self.features = nn.Sequential(
#                 nn.Linear(self._observation_shape[0], hidden_dim),
#                 activation,
#                 nn.Linear(hidden_dim, hidden_dim),
#                 activation,
#             )
#         else:
#             self.features = nn.Sequential(
#                 nn.Conv2d(self._observation_shape[0], 8, kernel_size=4, stride=2),
#                 activation,
#                 nn.Conv2d(16, 8, kernel_size=5, stride=1),
#                 activation,
#                 nn.Conv2d(16, 8, kernel_size=3, stride=1),
#                 activation,
#             )
        
#         self.fc = nn.Sequential(
#             nn.Linear(self._feature_size(), hidden_dim),
#             activation,
#             nn.Linear(hidden_dim, self._action_shape)
#         )

#     def net(self, x):
#         x = self.features(x)
#         x = self.flatten(x)
#         advantage = self.advantage(x)
#         value = self.value(x)
#         return value + advantage - advantage.mean(1, keepdim=True)

class ParallelDQN(DQNBase):
    """ DQN for parallel env sampling

    :param env: env object
    :type env: object
    :param net_args: network architecture arguments
    :type net_args: dict
    :param number_envs: number of environments
    :type number_envs: int
    :param kwargs: arbitrary keyword arguments.
    :type kwargs: dict
    """
    def __init__(self, env, net_args, number_envs, **kwargs):
        super(ParallelDQN, self).__init__(env, net_args, **kwargs)
        self.number_envs = number_envs

    def choose_action(self, state, epsilon):
        """Choose action acoording to state.

        :param state: state/observation input
        :type state:  torch.Tensor
        :param epsilon: epsilon for epsilon-greedy, defaults to 0.
        :type epsilon: float, optional
        :return: action
        :rtype: int or np.ndarray
        """  
        if random.random() > epsilon:  # NoisyNet does not use e-greedy
            with torch.no_grad():
                q_value = self.net(state)
                action = q_value.max(1)[1].detach().cpu().numpy()
        else:
            action = np.random.randint(self._action_shape, size=self.number_envs)
        return action

class ParallelDuelingDQN(DuelingDQN, ParallelDQN):
    """ DuelingDQN for parallel env sampling

    :param env: env object
    :type env: object
    :param net_args: network architecture arguments
    :type net_args: dict
    :param number_envs: number of environments
    :type number_envs: int
    :param kwargs: other arguments
    :type kwargs: dict

    Note: for mulitple inheritance, see a minimal example:

    .. code-block:: python

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
    def __init__(self, env, net_args, number_envs):
        super(ParallelDuelingDQN, self).__init__(env=env, net_args=net_args, number_envs=number_envs)

