import gym
import torch.nn as nn

class Agent(nn.Module):
    """
    A standard agent class.
    """
    def __init__(self, env):
        super(Agent, self).__init__()
        self.observation_shape = env.observation_space.shape
        self.action_shape = env.action_space.shape or env.action_space.n
        print(f"observation shape: {self.observation_shape}, action shape: {self.action_shape}")
    
    def choose_action(self, state, args):
        pass
    
    def save_model(self, path=None):
        pass

    def load_model(self, path=None):
        pass


class MultiAgent():
    """
    A class containing all agents in a game.
    """
    def __init__(self, env, agents):
        super(MultiAgent, self).__init__()
        # self.observation_spaces = env.observation_spaces
        # self.action_spaces = env.action_spaces
        self.agents = agents

    def choose_action(self, states):
        actions = []
        for state, agent in zip(states, self.agents): 
            action = agent.choose_action(state)
            actions.append(action)
        return actions
    
    def save_model(self, path=None):
        for agent in self.agents:
            agent.save_model(path)

    def load_model(self, path=None):
        for agent in self.agents:
            agent.load_model(path)



    
        
