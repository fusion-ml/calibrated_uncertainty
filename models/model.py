import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.nn.utils import weight_norm
from torch.autograd import Variable


class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features,
                 bias=True, use_bn=False,
                 actv_type='relu'):
        super(LinearLayer, self).__init__()

        """ linear layer """
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        """ batch normalization """
        if use_bn:
            self.bn = nn.BatchNorm1d(self.out_features)
        else:
            self.bn = None

        """ activation """
        if actv_type is None:
            self.activation = None
        elif actv_type == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif actv_type == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif actv_type == 'selu':
            self.activation = nn.SELU(inplace=True)
        else:
            raise ValueError

    def reset_parameters(self, reset_indv_bias=None):
        # init.kaiming_uniform_(self.weight, a=math.sqrt(0)) # kaiming init
        if (reset_indv_bias is None) or (reset_indv_bias is False):
            init.xavier_uniform_(self.weight, gain=1.0)  # xavier init
        if (reset_indv_bias is None) or ((self.bias is not None) and reset_indv_bias is True):
            init.constant_(self.bias, 0)

    def forward(self, input):
        out = F.linear(input, self.weight, self.bias)
        if self.bn:
            out = self.bn(out)
        if self.activation:
            out = self.activation(out)

        return out


class vanilla_nn(nn.Module):
    def __init__(self, input_size=1, output_size=1, bias=True,
                 hidden_size=400, num_layers=4,
                 use_bn=False, actv_type='relu',
                 softmax=False):

        super(vanilla_nn, self).__init__()
        self.softmax = softmax
        self.loss = nn.MSELoss()

        self.fcs = nn.ModuleList()
        self.fcs.append(LinearLayer(input_size, hidden_size, bias,
                                    use_bn=use_bn, actv_type=actv_type))
        for _ in range(num_layers-2):
            self.fcs.append(LinearLayer(hidden_size, hidden_size, bias,
                                        use_bn=use_bn, actv_type=actv_type))
        self.fcs.append(LinearLayer(hidden_size, output_size, bias,
                                    use_bn=False, actv_type=None))

    def forward(self, X):
        for layer in self.fcs:
            X = layer(X)

        if self.softmax:
            out = F.softmax(X, dim=1)
        else:
            out = X

        return out
