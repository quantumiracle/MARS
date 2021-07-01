import math
import torch.optim as optim

def choose_optimizer(name):
    if name == 'adam':
        return optim.Adam

    elif name == 'sgd':
        return optim.SGD

    # TODO: add more 
    else:
        raise NotImplementedError

class EpsilonScheduler():
    def __init__(self, eps_start, eps_final, eps_decay):
        self.eps_start = eps_start
        self.eps_final = eps_final
        self.eps_decay = eps_decay
        self.epsilon = self.eps_start

    def step(self, frame_idx):
        self.epsilon = self.eps_final + (self.eps_start - self.eps_final) * math.exp(-1. * frame_idx / self.eps_decay)
    
    def get_epsilon(self):
        return self.epsilon