import torch
import torch.optim as optim

def choose_optimizer(name):
    if name == 'adam':
        return optim.Adam

    elif name == 'sgd':
        return optim.SGD

    # TODO: add more 
    else:
        raise NotImplementedError