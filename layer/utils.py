import torch
from torch import nn
import torch.nn.functional as F

import math
import numbers

class LayerNorm(nn.Module):
    def __init__(self, dim, affine_shape=None, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.dim = dim
        
        self.eps = eps

        if affine_shape is not None:
            if isinstance(affine_shape, numbers.Integral):
                affine_shape = (affine_shape,)
            affine_shape = tuple(affine_shape)

            self.affine = True
            self.weight = nn.Parameter(torch.empty(*affine_shape))
            self.bias = nn.Parameter(torch.empty(*affine_shape))
        else:
            self.affine = False
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self._reset_parameters()

    def _reset_parameters(self):
        if self.affine:
            nn.init.constant_(self.weight, 0.1)
            nn.init.constant_(self.bias, 0)

    def forward(self, x):
        mean = x.mean(self.dim, keepdim=True)
        centered_x = x - mean
        var = (centered_x ** 2).mean(self.dim, keepdim=True)

        y = centered_x / (var + self.eps).sqrt()
        if self.affine:
            y = self.weight * y + self.bias

        return y

class Affine(nn.Module):
    def __init__(self, out_neurons, init_gamma=0.1, init_beta=0, eps=1e-5):
        super(Affine, self).__init__()
        if isinstance(out_neurons, numbers.Integral):
            out_neurons = (out_neurons,)
        out_neurons = tuple(out_neurons)

        self.gamma = nn.Parameter(torch.empty(*out_neurons))
        self.beta = nn.Parameter(torch.empty(*out_neurons))

        self._reset_parameters(init_gamma, init_beta)

    def _reset_parameters(self, init_gamma, init_beta):
        nn.init.constant_(self.gamma, init_gamma)
        nn.init.constant_(self.beta, init_beta)

    def forward(self, x):
        y = self.gamma * x + self.beta
        return y

class LinearLN(nn.Module):
    def __init__(self, in_neurons, out_neurons):
        super(LinearLN, self).__init__()

        self.weight = nn.Parameter(torch.empty(out_neurons, in_neurons))
        self.norm = nn.LayerNorm(out_neurons)
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # stdv = 1. / math.sqrt(self.weight.size(1))        
        # self.weight.data.uniform_(-stdv, stdv)

        nn.init.constant_(self.norm.weight, 0.1)
        nn.init.constant_(self.norm.bias, 0)

    def forward(self, x):
        y = F.linear(x, self.weight)
        return self.norm(y)
