import torch
import torch.nn as nn

class Agent(object):
    """
    A standard agent class.
    """
    def __init__(self, env, args):
        super(Agent, self).__init__()
        self.batch_size = args.batch_size
        if args.device == 'gpu':
            self.device =  torch.device("cuda:0")
        elif args.device == 'cpu':
            self.device = torch.device("cpu") 

    def choose_action(self, state, args):
        pass

    def store(self):
        """ Store a sample for either on-policy or off-policy algorithms."""
        pass

    def update(self):
        """ Update the agent. """
        pass

    def update_target(current_model, target_model):
        """
        Update the target model when necessary.
        """
        target_model.load_state_dict(current_model.state_dict())
        
    def save_model(self, path=None):
        pass

    def load_model(self, path=None):
        pass


class MultiAgent(Agent):
    """
    A class containing all agents in a game.
    """
    def __init__(self, env, agents, args):
        super(MultiAgent, self).__init__(env, args)
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



    
        
