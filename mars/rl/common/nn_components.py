import torch.nn as nn
import torch.nn.functional as F
import torch


class cReLU(nn.Module):
    """
    Note: for cReLU activation function, it doubles the output channels.
    """
    def forward(self, x):
        return torch.cat((F.relu(x), F.relu(-x)), dim=1)

class Flatten(nn.Module):
    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)

dSiLU = lambda x: torch.sigmoid(x)*(1+x*(1-torch.sigmoid(x)))
SiLU = lambda x: x*torch.sigmoid(x)


DimensionReductionActivations = ['Softmax', ]  # TODO add other activation functions requiring dimension reduction