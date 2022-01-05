import numpy as np
from mars.rl.agents.agent import Agent

class MetaLearner(Agent):
    def __init__(self,):
        self.models_family = []
        self.saved_checkpoints = []
        self.meta_strategy = None
        
    def step(self, model, *kargs):
        if len(self.saved_checkpoints) > 0:
            self._replace_with_meta(self.saved_checkpoints) 
    
    def choose_action(self, state):
        action = self.model.choose_action(state)
        return action

    def _replace_with_meta(self,checkpoints_to_replace_from, postfix=''):
        """ sample from the policy family according to meta strategy distribution, and replace certain agent"""
        sample_hist = np.random.multinomial(1, self.meta_strategy)  # meta nash policy is a distribution over the policy set, sample one policy from it according to meta nash for each episode
        policy_idx = np.squeeze(np.where(sample_hist>0))
        self.model.load_model(self.model_path+checkpoints_to_replace_from[policy_idx]+postfix)

    def _replace_agent_with_meta(self, model, agent_to_replace, checkpoints_to_replace_from, postfix=''):
        """ sample from the policy family according to meta strategy distribution, and replace certain agent"""
        sample_hist = np.random.multinomial(1, self.meta_strategy)  # meta nash policy is a distribution over the policy set, sample one policy from it according to meta nash for each episode
        policy_idx = np.squeeze(np.where(sample_hist>0))
        model.agents[agent_to_replace].load_model(self.model_path+checkpoints_to_replace_from[policy_idx]+postfix)

    def save_model(self, path: str = None):
        if len(self.saved_checkpoints) > 0:  # methods with family of models
            with open(self.model_path+'meta_strategy.npy', 'wb') as f:
                np.save(f, self.meta_strategy)
            with open(self.model_path+'policy_checkpoints.npy', 'wb') as f:
                np.save(f, self.saved_checkpoints)

    def load_model(self, model, path: str = None):
        self.model = model
        if path is not None:
            self.model_path = path
        with open(self.model_path+'meta_strategy.npy', 'rb') as f:
            self.meta_strategy = np.load(f)
        with open(self.model_path+'policy_checkpoints.npy', 'rb') as f:
            self.saved_checkpoints = np.load(f)
