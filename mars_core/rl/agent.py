import torch
import torch.nn as nn
import numpy as np
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
        self.not_learnable = False # whether the model is fixed (not learnable) or not

    def fix(self,):
        self.not_learnable = True

    def choose_action(self, state, *args):
        pass

    def scheduler_step(self, frame):
        """ Learning rate scheduler, epsilon scheduler, etc"""
        for scheduler in self.schedulers:
            scheduler.step(frame)

    def store(self, *args):
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
        
    def save_model(self, path=None):
        pass

    def load_model(self, path=None):
        pass

    @property
    def ready_to_update(self):
        pass


class MultiAgent(Agent):
    """
    A class containing all agents in a game.
    """
    def __init__(self, env, agents, args):
        super(MultiAgent, self).__init__(env, args)
        self.agents = agents
        not_learnable_list = []
        for i, agent in enumerate(agents):
            if agent.not_learnable:
                not_learnable_list.append(i)
        if len(not_learnable_list) < 1:
            prefix = 'No agent'

        else:
            prefix = f'Agents No. {not_learnable_list}'
        print(prefix+" are not learnable.")

    def choose_action(self, states):
        actions = []
        for state, agent in zip(states, self.agents): 
            action = agent.choose_action(state)
            actions.append(action)
        return actions

    def scheduler_step(self, frame):
        for agent in self.agents:
            agent.scheduler_step(frame)

    def store(self, sample):
        for agent, *s in zip(self.agents, *sample):
            agent.store(tuple(s))

    def update(self):
        losses = []
        for agent in self.agents:
            losses.append(agent.update())
        return losses
    
    def save_model(self, path=None):
        for agent in self.agents:
            agent.save_model(path)

    def load_model(self, path=None):
        for agent in self.agents:
            agent.load_model(path)

    @property
    def ready_to_update(self):
        ready_state = []
        for agent in self.agents:
            ready_state.append(agent.ready_to_update)
        if np.all(ready_state):
            return True 
        else:
            return False



    
        
