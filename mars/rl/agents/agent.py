import torch
import torch.nn as nn
import numpy as np
import gym
from mars.utils.typing import List, Union, StateType, ActionType, SampleType, SamplesType, ConfigurationDict
class Agent(object):
    """
    A standard agent class.
    """
    def __init__(self, env, args: ConfigurationDict):
        super(Agent, self).__init__()
        if isinstance(env.observation_space, list):  # when using parallel envs
            self.observation_space = env.observation_space[0]
        else:
            self.observation_space = env.observation_space

        if isinstance(env.action_space, list):  # when using parallel envs
            if isinstance(env.action_space[0], gym.spaces.Box):
                self.policy_type = 'gaussian_policy'
                self.action_dim = env.action_space[0].shape[0]
            else:
                self.policy_type = 'discrete_policy'
                self.action_dim = env.action_space[0].n
            self.action_space = env.action_space[0]
            
        else:
            if isinstance(env.action_space, gym.spaces.Box):
                self.policy_type = 'gaussian_policy'
                self.action_dim = env.action_space.shape[0]
            else:
                self.policy_type = 'discrete_policy'
                self.action_dim = env.action_space.n
            self.action_space = env.action_space

        print(self.policy_type, self.action_dim, self.action_space)

        self.batch_size = args.batch_size
        self.schedulers = []
        if args.device == 'gpu':
            self.device = torch.device("cuda:0")  # TODO
        elif args.device == 'cpu':
            self.device = torch.device("cpu")
        self.not_learnable = False  # whether the model is fixed (not learnable) or not

    def fix(self, ):
        self.not_learnable = True

    def choose_action(
        self, 
        state: List[StateType], 
        *args,
        **kwargs
        ) -> List[ActionType]:
        pass

    def scheduler_step(
        self, 
        frame: int
        ) -> None:
        """ Learning rate scheduler, epsilon scheduler, etc"""
        for scheduler in self.schedulers:
            scheduler.step(frame)

    def store(
        self, 
        sample: SampleType, 
        *args) -> None:
        """ Store a sample for either on-policy or off-policy algorithms."""
        pass

    def update(self):
        """ Update the agent. """
        pass

    def update_target(self, current_model, target_model):
        """
        Update the target model when necessary.
        """
        if isinstance(current_model, list) and isinstance(target_model, list):
            for cur_m, tar_m in zip(current_model, target_model):
                tar_m.load_state_dict(cur_m.state_dict())
        else:
            target_model.load_state_dict(current_model.state_dict())

    def save_model(self, path: str = None, *args, **kwargs):
        pass

    def load_model(self, path: str = None, *args, **kwargs):
        pass

    @property
    def ready_to_update(self) -> bool:
        """ A function return whether the agent is ready to be updated.
        """
        return True

