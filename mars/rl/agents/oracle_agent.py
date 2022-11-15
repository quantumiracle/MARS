import torch
import torch.nn as nn
import numpy as np
from mars.utils.typing import List, Union, StateType, ActionType, SampleType, SamplesType, ConfigurationDict
from .agent import Agent

class OracleAgent(Agent):
    """
    A standard agent class.
    """
    def __init__(self, env, args: ConfigurationDict):
        super().__init__(env, args)
        self.policies = np.load('data/nash_dqn_test/oracle_nash.npy', allow_pickle=True)
        self.agent_idx = 0 # 0-th agent to be exploited
        self.policy = np.array(self.policies)[:,:, self.agent_idx]  # (#transitions, #states, #actions)
        self.policy = self.policy.reshape(-1, env.action_space.n)

    def choose_action(
        self, 
        state: List[StateType], 
        *args,
        **kwargs
        ) -> List[ActionType]:
        dist = self.policy[state[0][0]]
        sample_hist = np.random.multinomial(1, dist)
        action = np.where(sample_hist>0)
        return action[0]
