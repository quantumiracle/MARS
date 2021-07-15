import torch.nn as nn
import torch.nn.functional as F
import torch
import copy
from .nn_components import cReLU, Flatten


class NetBase(nn.Module):
    """ Base network class for policy/value function """
    def __init__(self, input_space, output_space):
        super(NetBase, self).__init__()
        self._preprocess(input_space, output_space)
        self.features = None
        self.body = None

    def _preprocess(self, input_space, output_space):
        """
        In general RL setting, 
        input_space: observation_space
        output_space: action_space
        """
        # if pass in list of spaces, then take the first one
        if isinstance(input_space, list):
            input_space = input_space[0]
        if isinstance(output_space, list):
            output_space = output_space[0]     
               
        self._observation_space = input_space
        self._observation_shape = input_space.shape
        if len(self._observation_shape) == 1:
            self._observation_dim = self._observation_shape[0]
        else:  # high-dim state
            pass

        self._action_space = output_space
        self._action_shape = output_space.shape or output_space.n
        if isinstance(self._action_shape, int):  # Discrete space
            self._action_dim = self._action_shape
        else:
            self._action_dim = self._action_shape[0]
        # print(f"observation shape: {self._observation_shape}, action shape: {self._action_shape}")
    
    def _get_output_dim(self, model_for):  
        if model_for == 'gaussian_policy':  # diagonal Gaussian: means and stds
            return 2*self._action_dim 
        elif model_for == 'discrete_policy':  # categorical
            return self._action_dim
        elif model_for == 'discrete_q': 
            return self._action_dim
        elif model_for == 'continuous_q' or 'value':
            return 1

    def _construct_net(self, args):
        pass

    def forward(self, x):
        """ need to be overwritten by the subclass """
        if self.features is not None:
            x = self.features(x)
        x = self.body(x)
        return x

    def _feature_size(self):
        if isinstance(self._observation_shape, int):
            return self.features(torch.zeros(1, self._observation_shape)).view(
                1, -1).size(1)
        else:
            return self.features(torch.zeros(1,
                                             *self._observation_shape)).view(
                                                 1, -1).size(1)


class MLP(NetBase):
    def __init__(self, input_space, output_space, net_args, model_for):
        super().__init__(input_space, output_space)
        layers_config = copy.deepcopy(net_args)
        layers_config['hidden_dim_list'].insert(0, self._observation_dim)
        output_dim = self._get_output_dim(model_for)
        layers_config['hidden_dim_list'].append(output_dim)
        self.features = None
        self.body = self._construct_net(layers_config)

    def _construct_net(self, layers_config):
        layers = []
        for j in range(len(layers_config['hidden_dim_list']) - 1):
            tmp = [
                nn.Linear(layers_config['hidden_dim_list'][j],
                          layers_config['hidden_dim_list'][j + 1])
            ]
            if layers_config['hidden_activation']:
                tmp.append(
                    getattr(torch.nn, layers_config['hidden_activation'])())
            layers += tmp
        if layers_config['output_activation']:
            layers += [getattr(torch.nn, layers_config['output_activation'])()]
        return nn.Sequential(*layers)


class CNN(NetBase):
    def __init__(self, input_space, output_space, net_args, model_for):
        super().__init__(input_space, output_space)
        layers_config = copy.deepcopy(net_args)
        layers_config['channel_list'].insert(0, self._observation_shape[0])
        self.features = self._construct_cnn_net(layers_config)
        layers_config['hidden_dim_list'].insert(0, self._feature_size())
        output_dim = self._get_output_dim(model_for)
        layers_config['hidden_dim_list'].append(output_dim)
        self.body = self._construct_net(layers_config)

    def _construct_cnn_net(self, layers_config):
        layers = []
        for i in range(len(layers_config["channel_list"] - 1)):
            tmp = [
                nn.Conv2d(layers_config["channel_list"][i],
                          layers_config["channel_list"][i + 1])
            ]
            if layers_config['hidden_activation']:
                tmp.append(
                    getattr(torch.nn, layers_config["hidden_activation"])())
            # TODO add pooling
            layers += tmp
        return nn.Sequential(*layers)

    def _construct_net(self, layers_config):
        layers = []
        layers.append(Flatten())
        for j in range(len(layers_config['hidden_dim_list']) - 1):
            tmp = [
                nn.Linear(layers_config['hidden_dim_list'][j],
                          layers_config['hidden_dim_list'][j + 1])
            ]
            if layers_config['hidden_activation']:
                tmp.append(
                    getattr(torch.nn, layers_config['hidden_activation'])())
            layers += tmp
        if layers_config['output_activation']:
            layers += [getattr(torch.nn, layers_config['output_activation'])()]
        return nn.Sequential(*layers)


def get_model(model_type="mlp"):
    """
        :param str model_for: 'value' or 'policy'
    """
    if model_type == "mlp":
        handler = MLP
    elif model_type == "rnn":
        handler = NotImplementedError
    elif model_type == "cnn":
        handler = CNN
    elif model_type == "rcnn":
        raise NotImplementedError
    else:
        raise NotImplementedError

    def builder(input_space, output_space, net_args, model_for):
        model = handler(input_space, output_space, copy.deepcopy(net_args), model_for)
        return model

    return builder


# class PolicyMLP(NetBase):
#     def __init__(self, state_space, action_space, hidden_dim, device):
#         super().__init__( state_space, action_space)
#         self.fc1   = nn.Linear(self._observation_dim, hidden_dim)
#         self.fc2   = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3   = nn.Linear(hidden_dim, self._action_dim)
#         self.device = device

#     def forward(self, x, softmax_dim = -1):
#         # x = torch.FloatTensor(x).unsqueeze(0)
#         x = F.tanh(self.fc1(x))
#         x = F.tanh(self.fc2(x))
#         x = self.fc3(x)
#         prob = F.softmax(x, dim=softmax_dim)
#         return prob

# class ValueMLP(NetBase):
#     def __init__(self, state_space, hidden_dim, init_w=3e-3):
#         super().__init__(state_space)

#         self.linear1 = nn.Linear(self._observation_dim, hidden_dim)
#         self.linear2 = nn.Linear(hidden_dim, hidden_dim)
#         # self.linear3 = nn.Linear(hidden_dim, hidden_dim)
#         self.linear4 = nn.Linear(hidden_dim, 1)

#     def forward(self, state):
#         x = F.tanh(self.linear1(state))
#         x = F.tanh(self.linear2(x))
#         # x = F.relu(self.linear3(x))
#         x = self.linear4(x)
#         return x

# dSiLU = lambda x: torch.sigmoid(x)*(1+x*(1-torch.sigmoid(x)))
# SiLU = lambda x: x*torch.sigmoid(x)

# class PolicyCNN(NetBase):
#     def __init__(self, state_space, action_space, hidden_dim, device):
#         super().__init__(state_space, action_space)
#         X_channel = self._observation_space.shape[2]
#         X_dim = self._observation_space.shape[1]
#         assert self._observation_space.shape[0] == self._observation_space.shape[1]
#         self.CONV_NUM_FEATURE_MAP=8
#         self.CONV_KERNEL_SIZE=4
#         self.CONV_STRIDE=2
#         self.CONV_PADDING=1
#         self.in_layer = nn.Sequential(
#             nn.Conv2d(X_channel, self.CONV_NUM_FEATURE_MAP, self.CONV_KERNEL_SIZE, self.CONV_STRIDE, self.CONV_PADDING, bias=False),  # in_channels, out_channels, kernel_size, stride=1, padding=0
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(self.CONV_NUM_FEATURE_MAP, self.CONV_NUM_FEATURE_MAP * 2, self.CONV_KERNEL_SIZE, self.CONV_STRIDE, self.CONV_PADDING, bias=False),
#             nn.BatchNorm2d(self.CONV_NUM_FEATURE_MAP * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#         )
#         conv1_size = int((X_dim-self.CONV_KERNEL_SIZE+2*self.CONV_PADDING)/self.CONV_STRIDE) + 1
#         conv2_size = int((conv1_size-self.CONV_KERNEL_SIZE+2*self.CONV_PADDING)/self.CONV_STRIDE) + 1
#         in_layer_dim = int(self.CONV_NUM_FEATURE_MAP*2* (conv2_size)**2)
#         self.fc_h1 = nn.Linear(in_layer_dim, hidden_dim)
#         self.fc_pi = nn.Linear(hidden_dim, self._action_dim)

#     def forward(self, x, softmax_dim = -1):
#         x = x.permute(0,3,1,2)  # change from (N,H,W,C) to (N,C,H,W) for torch
#         if len(x.shape) >1:
#             if len(x.shape) ==3:
#                 x = x.unsqueeze(0)
#             x = SiLU(self.in_layer(x))
#             x = x.reshape(x.shape[0], -1)
#         else:
#             x = F.relu(self.in_layer(x))
#         x = dSiLU(self.fc_h1(x))
#         x = self.fc_pi(x)
#         prob = F.softmax(x, dim=softmax_dim)
#         return prob

# class ValueCNN(NetBase):
#     def __init__(self, state_space, hidden_dim):
#         super().__init__(state_space)
#         X_channel = self._observation_space.shape[2]
#         X_dim = self._observation_space.shape[1]
#         assert self._observation_space.shape[0] == self._observation_space.shape[1]
#         self.CONV_NUM_FEATURE_MAP=8
#         self.CONV_KERNEL_SIZE=4
#         self.CONV_STRIDE=2
#         self.CONV_PADDING=1
#         self.in_layer = nn.Sequential(
#             nn.Conv2d(X_channel, self.CONV_NUM_FEATURE_MAP, self.CONV_KERNEL_SIZE, self.CONV_STRIDE, self.CONV_PADDING, bias=False),  # in_channels, out_channels, kernel_size, stride=1, padding=0
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(self.CONV_NUM_FEATURE_MAP, self.CONV_NUM_FEATURE_MAP * 2, self.CONV_KERNEL_SIZE, self.CONV_STRIDE, self.CONV_PADDING, bias=False),
#             nn.BatchNorm2d(self.CONV_NUM_FEATURE_MAP * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#         )
#         conv1_size = int((X_dim-self.CONV_KERNEL_SIZE+2*self.CONV_PADDING)/self.CONV_STRIDE) + 1
#         conv2_size = int((conv1_size-self.CONV_KERNEL_SIZE+2*self.CONV_PADDING)/self.CONV_STRIDE) + 1
#         in_layer_dim = int(self.CONV_NUM_FEATURE_MAP*2* (conv2_size)**2)
#         self.fc_h1 = nn.Linear(in_layer_dim, hidden_dim)
#         self.fc_v = nn.Linear(hidden_dim, 1)

#     def forward(self, x, softmax_dim = -1):
#         x = x.permute(0,3,1,2)  # change from (N,H,W,C) to (N,C,H,W) for torch
#         if len(x.shape) >1:
#             if len(x.shape) ==3:
#                 x = x.unsqueeze(0)
#             x = SiLU(self.in_layer(x))
#             x = x.reshape(x.shape[0], -1)
#         else:
#             x = F.relu(self.in_layer(x))
#         x = dSiLU(self.fc_h1(x))
#         x = self.fc_v(x)
#         return x