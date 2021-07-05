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

    Definition of 'not_learnable': the agent is not self-updating using
    the RL loss, it's either never updated (i.e., 'fixed') or updated as
    a delayed copy of other learnable agents with the MARL learning scheme.
    """
    def __init__(self, env, agents, args):
        super(MultiAgent, self).__init__(env, args)
        self.agents = agents
        self.number_of_agents = len(self.agents)
        self.not_learnable_list = []
        for i, agent in enumerate(agents):
            if agent.not_learnable or \
                (args.marl_method is not None and i != args.marl_spec['trainable_agent_idx']):
                self.not_learnable_list.append(i)
        if len(self.not_learnable_list) < 1:
            prefix = 'No agent'

        else:
            prefix = f'Agents No. {self.not_learnable_list} (index starting from 0)'
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
        for i, agent, *s in zip(np.arange(self.number_of_agents), self.agents, *sample):
            if i not in self.not_learnable_list: # no need to store samples for not learnable models
                agent.store(tuple(s))

    def update(self):
        losses = []
        for i, agent in enumerate(self.agents):
            if i not in self.not_learnable_list:
                losses.append(agent.update())
            else:
                losses.append(0.)
        return losses
    
    def save_model(self, path=None):
        for agent in self.agents:
            agent.save_model(path)

    def load_model(self, path=None):
        for agent in self.agents:
            agent.load_model(path)

    @property
    def ready_to_update(self) -> bool:
        """ 
        A function returns whether all learnable agents are ready to be updated, 
        called from the main rollout function.
        """
        ready_state = []
        for i, agent in enumerate(self.agents):
            if i not in self.not_learnable_list:
                ready_state.append(agent.ready_to_update)
        if np.all(ready_state):
            return True 
        else:
            return False



    
        
