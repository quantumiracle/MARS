import torch
import torch.nn as nn
import numpy as np
from mars.utils.typing import List, Union, StateType, ActionType, SampleType, SamplesType, ConfigurationDict
class Agent(object):
    """
    A standard agent class.
    """
    def __init__(self, env, args: ConfigurationDict):
        super(Agent, self).__init__()
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

