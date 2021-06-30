import torch.nn as nn
import torch.nn.functional as F
import torch

class NetBase(nn.Module):
    """ Base network class for policy/value function """
    def __init__(self, state_space, action_space=None):
        super(NetBase, self).__init__()
        self._state_space = state_space
        self._state_shape = state_space.shape
        if len(self._state_shape) == 1:
            self._state_dim = self._state_shape[0]
        else:  # high-dim state
            pass  
        if action_space is not None:
            self._action_space = action_space
            self._action_shape = action_space.shape
            if len(self._action_shape) < 1:  # Discrete space
                self._action_dim = action_space.n
            else:
                self._action_dim = self._action_shape[0]

    def forward(self,):
        """ need to be overwritten by the subclass """
        raise NotImplementedError


class PolicyMLP(NetBase):
    def __init__(self, state_space, action_space, hidden_dim, device):
        super().__init__( state_space, action_space)
        self.fc1   = nn.Linear(self._state_dim, hidden_dim)
        self.fc2   = nn.Linear(hidden_dim, hidden_dim)
        self.fc3   = nn.Linear(hidden_dim, self._action_dim)
        self.device = device

    def forward(self, x, softmax_dim = -1):
        # x = torch.FloatTensor(x).unsqueeze(0)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

class ValueMLP(NetBase):
    def __init__(self, state_space, hidden_dim, init_w=3e-3):
        super().__init__(state_space)
        
        self.linear1 = nn.Linear(self._state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        # self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)

        
    def forward(self, state):
        x = F.tanh(self.linear1(state))
        x = F.tanh(self.linear2(x))
        # x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x


dSiLU = lambda x: torch.sigmoid(x)*(1+x*(1-torch.sigmoid(x)))
SiLU = lambda x: x*torch.sigmoid(x)

class PolicyCNN(NetBase):
    def __init__(self, state_space, action_space, hidden_dim, device):
        super().__init__(state_space, action_space)
        X_channel = self._state_space.shape[2]
        X_dim = self._state_space.shape[1]
        assert self._state_space.shape[0] == self._state_space.shape[1]
        self.CONV_NUM_FEATURE_MAP=8
        self.CONV_KERNEL_SIZE=4
        self.CONV_STRIDE=2
        self.CONV_PADDING=1
        self.in_layer = nn.Sequential(
            nn.Conv2d(X_channel, self.CONV_NUM_FEATURE_MAP, self.CONV_KERNEL_SIZE, self.CONV_STRIDE, self.CONV_PADDING, bias=False),  # in_channels, out_channels, kernel_size, stride=1, padding=0
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.CONV_NUM_FEATURE_MAP, self.CONV_NUM_FEATURE_MAP * 2, self.CONV_KERNEL_SIZE, self.CONV_STRIDE, self.CONV_PADDING, bias=False),
            nn.BatchNorm2d(self.CONV_NUM_FEATURE_MAP * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        conv1_size = int((X_dim-self.CONV_KERNEL_SIZE+2*self.CONV_PADDING)/self.CONV_STRIDE) + 1
        conv2_size = int((conv1_size-self.CONV_KERNEL_SIZE+2*self.CONV_PADDING)/self.CONV_STRIDE) + 1
        in_layer_dim = int(self.CONV_NUM_FEATURE_MAP*2* (conv2_size)**2)
        self.fc_h1 = nn.Linear(in_layer_dim, hidden_dim)  
        self.fc_pi = nn.Linear(hidden_dim, self._action_dim)  


    def forward(self, x, softmax_dim = -1):
        x = x.permute(0,3,1,2)  # change from (N,H,W,C) to (N,C,H,W) for torch
        if len(x.shape) >1:
            if len(x.shape) ==3:
                x = x.unsqueeze(0)
            x = SiLU(self.in_layer(x))
            x = x.reshape(x.shape[0], -1)
        else:
            x = F.relu(self.in_layer(x))
        x = dSiLU(self.fc_h1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

class ValueCNN(NetBase):
    def __init__(self, state_space, hidden_dim):
        super().__init__(state_space)
        X_channel = self._state_space.shape[2]
        X_dim = self._state_space.shape[1]
        assert self._state_space.shape[0] == self._state_space.shape[1]
        self.CONV_NUM_FEATURE_MAP=8
        self.CONV_KERNEL_SIZE=4
        self.CONV_STRIDE=2
        self.CONV_PADDING=1
        self.in_layer = nn.Sequential(
            nn.Conv2d(X_channel, self.CONV_NUM_FEATURE_MAP, self.CONV_KERNEL_SIZE, self.CONV_STRIDE, self.CONV_PADDING, bias=False),  # in_channels, out_channels, kernel_size, stride=1, padding=0
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.CONV_NUM_FEATURE_MAP, self.CONV_NUM_FEATURE_MAP * 2, self.CONV_KERNEL_SIZE, self.CONV_STRIDE, self.CONV_PADDING, bias=False),
            nn.BatchNorm2d(self.CONV_NUM_FEATURE_MAP * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        conv1_size = int((X_dim-self.CONV_KERNEL_SIZE+2*self.CONV_PADDING)/self.CONV_STRIDE) + 1
        conv2_size = int((conv1_size-self.CONV_KERNEL_SIZE+2*self.CONV_PADDING)/self.CONV_STRIDE) + 1
        in_layer_dim = int(self.CONV_NUM_FEATURE_MAP*2* (conv2_size)**2)
        self.fc_h1 = nn.Linear(in_layer_dim, hidden_dim)  
        self.fc_v = nn.Linear(hidden_dim, 1)  


    def forward(self, x, softmax_dim = -1):
        x = x.permute(0,3,1,2)  # change from (N,H,W,C) to (N,C,H,W) for torch
        if len(x.shape) >1:
            if len(x.shape) ==3:
                x = x.unsqueeze(0)
            x = SiLU(self.in_layer(x))
            x = x.reshape(x.shape[0], -1)
        else:
            x = F.relu(self.in_layer(x))
        x = dSiLU(self.fc_h1(x))
        x = self.fc_v(x)
        return x