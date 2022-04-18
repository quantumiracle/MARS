import math
import torch.optim as optim

def choose_optimizer(name):
    """Select an optimizer.

    :param name: optimizer type
    :type name: str
    :raises NotImplementedError: optimizer type not found
    :return: the optimizer
    :rtype: class
    """
    if name == 'adam':
        return optim.Adam

    elif name == 'sgd':
        return optim.SGD

    # TODO: add more 
    else:
        raise NotImplementedError

class EpsilonScheduler():
    def __init__(self, eps_start, eps_final, eps_decay):
        """A scheduler for epsilon-greedy strategy.

        :param eps_start: starting value of epsilon, default 1. as purely random policy 
        :type eps_start: float
        :param eps_final: final value of epsilon
        :type eps_final: float
        :param eps_decay: number of timesteps from eps_start to eps_final
        :type eps_decay: int
        """
        self.eps_start = eps_start
        self.eps_final = eps_final
        self.eps_decay = eps_decay
        self.epsilon = self.eps_start
        self.ini_frame_idx = 0
        self.current_frame_idx = 0
        self.restart_period = 300*10000 # periodically reset the scheduler

    def reset(self, ):
        """ Reset the scheduler """
        self.ini_frame_idx = self.current_frame_idx

    def step(self, frame_idx):
        """
        The choice of eps_decay:
        ------------------------
        start = 1
        final = 0.01
        decay = 10**6  # the decay steps can be 1/10 over all steps 10000*1000
        final + (start-final)*np.exp(-1*(10**7)/decay)
        
        => 0.01

        """
        self.current_frame_idx = frame_idx
        delta_frame_idx = self.current_frame_idx - self.ini_frame_idx
        self.epsilon = self.eps_final + (self.eps_start - self.eps_final) * math.exp(-1. * delta_frame_idx / self.eps_decay)
    
        # if delta_frame_idx > self.restart_period:
        #     self.reset()

    def get_epsilon(self):
        return self.epsilon
