import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import copy
from .nn_components import cReLU, Flatten, DimensionReductionActivations

class NetBase(nn.Module):
    """ Base network class for policy/value function """
    def __init__(self, input_space, output_space):
        super(NetBase, self).__init__()
        self._preprocess(input_space, output_space)
        # self.features = None
        self.body = None

    def features_net(self, x, *args):
        return None

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
        elif len(self._observation_shape) == 2:
            self._observation_dim = self._observation_shape[-1]
        else:  # high-dim state
            pass

        self._action_space = output_space
        try:
            self._action_shape = output_space.shape or output_space.n # depends on different gym versions, some higher version (like 0.19) can call .shape without throwing an error; some lower version (like 0.9) will throw error for this. 
        except:
            self._action_shape = output_space.n
        if isinstance(self._action_shape, int):  # Discrete space
            self._action_dim = self._action_shape
        else:
            self._action_dim = self._action_shape[0]
        # print(f"observation shape: {self._observation_shape}, action shape: {self._action_shape}")
    
    def _get_output_dim(self, model_for):  
        if model_for == 'gaussian_policy':  # diagonal Gaussian: means and stds
            return 2*self._action_dim 
        elif model_for == 'independent_gaussian_policy': # std is handled independently
            return self._action_dim  
        elif model_for == 'discrete_policy':  # categorical
            return self._action_dim
        elif model_for in ['discrete_q', 'feature']: 
            return self._action_dim
        elif model_for in ['continuous_q', 'value']:
            return 1
        else:
            return self._action_dim


    def _construct_net(self, args):
        pass

    def forward(self, x):
        """ need to be overwritten by the subclass """
        features = self.features_net(x)
        if features is not None:
            x = features
        x = self.body(x)
        return x

    def _feature_size(self):
        if isinstance(self._observation_shape, int):
            return self.features_net(torch.zeros(1, self._observation_shape)).view(
                1, -1).size(1)
        else:
            return self.features_net(torch.zeros(1,
                                             *self._observation_shape)).view(
                                                 1, -1).size(1)
    def _weight_init(self, m):
        if isinstance(m, nn.Linear):
            # Use torch default initialization for Linear here: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L44-L48
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(m.bias, -bound, bound)

        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            # nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.kaiming_uniform_(m.weight)
            nn.init.normal_(m.bias)
        # else:
        #     print(f"{m} is not initialized.")

    def reinit(self, ):
        """Reinitialize the parameters of a network.
        """
        self.apply(self._weight_init)


class MLP(NetBase):
    def __init__(self, input_space, output_space, net_args, model_for=None):
        super().__init__(input_space, output_space)
        layers_config = copy.deepcopy(net_args)
        layers_config['hidden_dim_list'].insert(0, self._observation_dim)
        output_dim = self._get_output_dim(model_for)
        layers_config['hidden_dim_list'].append(output_dim)
        self.body = self._construct_net(layers_config)

    def features_net(self, *args):
        return None

    def _construct_net(self, layers_config):
        layers = []
        for j in range(len(layers_config['hidden_dim_list']) - 1):
            tmp = [
                nn.Linear(layers_config['hidden_dim_list'][j],
                          layers_config['hidden_dim_list'][j + 1])
            ]
            if j < len(layers_config['hidden_dim_list']) - 2 and layers_config['hidden_activation']:  # hidden activation should not be added to last layer
                tmp.append(
                    _get_activation(layers_config['hidden_activation'])())
            layers += tmp
        if layers_config['output_activation']:
            if layers_config['output_activation'] in DimensionReductionActivations:  
                layers += [_get_activation(layers_config['output_activation'])(dim=-1)]  # dim=-1 is critical! otherwise may not report bug but hurt performance
            else:
                layers += [_get_activation(layers_config['output_activation'])()]
        return nn.Sequential(*layers)


class CNN(NetBase):
    def __init__(self, input_space, output_space, net_args, model_for=None):
        super().__init__(input_space, output_space)
        layers_config = copy.deepcopy(net_args)
        layers_config['channel_list'].insert(0, self._observation_shape[0])
        self.features = self._construct_cnn_net(layers_config)  # this need to be built and called in features_net() before self._feature_size()
        layers_config['hidden_dim_list'].insert(0, self._feature_size())
        output_dim = self._get_output_dim(model_for)
        layers_config['hidden_dim_list'].append(output_dim)
        self.body = self._construct_net(layers_config)

    def features_net(self, x, *args):
        return self.features(x)

    def _construct_cnn_net(self, layers_config):
        layers = []
        for i in range(len(layers_config["channel_list"]) - 1):
            tmp = [
                nn.Conv2d(layers_config["channel_list"][i],
                          layers_config["channel_list"][i + 1],
                          kernel_size = layers_config["kernel_size_list"][i],
                          stride = layers_config["stride_list"][i]
                          )
            ]
            if layers_config['hidden_activation']:
                tmp.append(
                    _get_activation(layers_config['hidden_activation'])())
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
            if j < len(layers_config['hidden_dim_list']) - 2 and layers_config['hidden_activation']:  # hidden activation should not be added to last layer
                tmp.append(_get_activation(layers_config['hidden_activation'])()
                    )
            layers += tmp
        if layers_config['output_activation']:
            if layers_config['output_activation'] in DimensionReductionActivations:
                layers += [_get_activation(layers_config['output_activation'])(dim=-1)]  # dim=-1 is critical! otherwise may not report bug but hurt performance
            else:
                layers += [_get_activation(layers_config['output_activation'])()]
        return nn.Sequential(*layers)

class ImpalaCNN(NetBase):
    """
    Model used in the paper "IMPALA: Scalable Distributed Deep-RL with
    Importance Weighted Actor-Learner Architectures" https://arxiv.org/abs/1802.01561
    """
    def __init__(self, input_space, output_space, net_args, model_for=None):
        super().__init__(input_space, output_space)
        self.ResidualRepeat = 2  # repeat residual blocks in one conv sequence

        self.layers_config = copy.deepcopy(net_args)
        self.layers_config['channel_list'].insert(0, self._observation_shape[0])

        cnn_layers, max_pool_layers = self._construct_cnn_layers(self.layers_config)
        self.cnn_layers = nn.ModuleList(cnn_layers)
        self.max_pool_layers = nn.ModuleList(max_pool_layers)
        self._construct_residual_blocks()

        self.layers_config['hidden_dim_list'].insert(0, self._feature_size())
        output_dim = self._get_output_dim(model_for)
        self.layers_config['hidden_dim_list'].append(output_dim)
        self.body = self._construct_net(self.layers_config)

    def features_net(self, x):
        for cnn_layer, max_pool_layer, residual_blocks in zip(self.cnn_layers, self.max_pool_layers, self.residual_blocks_whole):
            y = cnn_layer(x)
            # y = max_pool_layer(y)  # TODO: no 'same' padding in PyTorch, so not added here
            for i in range(self.ResidualRepeat):
                y = residual_blocks[i](y) + y

        return y
        
    def _construct_net(self, layers_config):
        layers = []
        layers.append(Flatten())
        for j in range(len(layers_config['hidden_dim_list']) - 1):
            tmp = [
                nn.Linear(layers_config['hidden_dim_list'][j],
                          layers_config['hidden_dim_list'][j + 1])
            ]
            if j < len(layers_config['hidden_dim_list']) - 2 and layers_config['hidden_activation']:  # hidden activation should not be added to last layer
                tmp.append(_get_activation(layers_config['hidden_activation'])()
                    )
            layers += tmp
        if layers_config['output_activation']:
            if layers_config['output_activation'] in DimensionReductionActivations:
                layers += [_get_activation(layers_config['output_activation'])(dim=-1)]  # dim=-1 is critical! otherwise may not report bug but hurt performance
            else:
                layers += [_get_activation(layers_config['output_activation'])()]
        return nn.Sequential(*layers)

    def _construct_cnn_layers(self, layers_config):
        cnn_layers = []
        max_pool_layers = []
        for i in range(len(layers_config["channel_list"]) - 1):
            cnn_layers.append(self._fixed_cnn_layer(layers_config["channel_list"][0]))
            max_pool_layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=2))
        return cnn_layers, max_pool_layers

    def _fixed_cnn_layer(self, channels, kernel_size=3, stride=1, padding='same'):
        return nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding)
    
    def _construct_residual_blocks(self):
        # use fixed channels according to https://github.com/openai/baselines/blob/master/baselines/common/models.py 
        self.residual_blocks_whole = nn.ModuleList([])
        for i in range(len(self.layers_config["channel_list"]) - 1):
            self.residual_blocks = nn.ModuleList([])
            for j in range(self.ResidualRepeat):
                tmp = [ _get_activation('ReLU')(),
                        self._fixed_cnn_layer(self.layers_config["channel_list"][0]),
                        _get_activation('ReLU')(),
                        self._fixed_cnn_layer(self.layers_config["channel_list"][0]),
                    ]

                self.residual_blocks.append(nn.Sequential(*tmp))
            self.residual_blocks_whole.append(self.residual_blocks)

def _get_activation(activation_type):
    """
    Get the activation function.
        :param str activation_type: like 'ReLU', 'LeakyReLU', 'CReLU', 'Softmax', etc
    """
    if activation_type == 'CReLU':  # Notice that cReLU will change output dimension
        return cReLU
    else:
        try:
            activation = getattr(torch.nn, activation_type)
            return activation
        except:
            print(f"Activation type {activation_type} not implemented.")


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
    elif model_type == "impala_cnn":
        handler = ImpalaCNN
    elif model_type == "rcnn":
        raise NotImplementedError
    else:
        raise NotImplementedError

    def builder(input_space, output_space, net_args, model_for):
        model = handler(input_space, output_space, copy.deepcopy(net_args), model_for)
        return model

    return builder

if __name__ == '__main__':
    from gym import spaces
    obs_space = spaces.Box(low=0, high=255, shape=(10,))
    act_space = spaces.Discrete(3)
    net_args = {'hidden_dim_list': [64, 64, 64],  
        'hidden_activation': 'ReLU',
        'output_activation': False}
    model = get_model('mlp')(obs_space, act_space, net_args, model_for='discrete_q')
    print(model)
    for p in model.parameters():
        print(p)

    # def weight_init(m):
    #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
    #         nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
    #         nn.init.normal_(m.bias)

    # model.apply(weight_init)
    model.reinit()
    for p in model.parameters():
        print(p)
