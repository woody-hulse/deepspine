import torch
from torch import nn
import torch.nn.functional as F
from .utils import LayerNorm, Affine

# class MLP(nn.Module):
#     def __init__(self, in_features, out_features, activation, interm=None):
#         super(MLP, self).__init__()

#         if activation == 'relu':
#             activation = nn.ReLU(inplace=True)
#         elif activation == 'tanh':
#             activation = nn.Tanh()
#         else:
#             activation = None

        
#         mlp = []
#         if interm is not None:
#             if interm['activation'] == 'relu':
#                 interm_activation = nn.ReLU(inplace=True)
#             elif interm['activation'] == 'tanh':
#                 interm_activation = nn.Tanh()
#             else:
#                 interm_activation = None
                
#             for interm_features in interm['features']:
#                 mlp.append(nn.Linear(in_features, interm_features))
#                 if interm_activation is not None:
#                     mlp.append(interm_activation)

#                 in_features = interm_features
        
#         mlp.append(nn.Linear(in_features, out_features))
#         if activation is not None:
#             mlp.append(activation)

#         self.mlp = nn.Sequential(*mlp)

#     def forward(self, x):
#         y = self.mlp(x)

#         return y

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups=1, bias=True, pad=0):
        super(CausalConv1d, self).__init__()
        self.pad = pad
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, groups=groups, bias=bias, padding=pad)
        
    def forward(self, x):
        y = self.conv(x)
        if self.pad > 0:
            y = y[:,:,:-self.pad]
        return y


class ConvNet1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups=1, use_norm=None, activation='relu'):
        super(ConvNet1d, self).__init__()

        if activation == 'relu':
            activation = nn.ReLU(inplace=True)
        elif activation == 'tanh':
            activation = nn.Tanh()
        elif activation == 'sigmoid':
            activation = nn.Sigmoid()
        else:
            activation = None
        
        self.activation = activation
        self.use_norm = use_norm
        self.groups = groups

        self.cnn = CausalConv1d(in_channels, out_channels, kernel_size=kernel_size, groups=groups, pad=kernel_size-1)

        if use_norm:
            self.norm = LayerNorm(dim=2, affine_shape=[groups, int(out_channels/groups), 1])

    def forward(self, x):
        y = self.cnn(x) # BxCxT
        
        if self.use_norm:
            batch_size = y.size(0)
            T = y.size(-1)

            y = y.view(batch_size, self.groups, -1, T)
            y = self.norm(y)
            y = y.view(batch_size, -1, T)
                
        if self.activation is not None:
            y = self.activation(y)
            
        return y

class SensoryEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups=1, use_norm=None, activation='relu'):
        super(SensoryEncoder, self).__init__()
        
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = None
        
        self.use_norm = use_norm
        self.groups = groups

        self.kinematic2afferent = ConvNet1d(in_channels, out_channels, kernel_size, groups, use_norm, activation)
        self.spike2rate = CausalConv1d(1, 1, kernel_size=30, groups=1, pad=29)
        self.antidromic = CausalConv1d(1, 1, kernel_size=20, groups=1, pad=19)   
        self.affine = Affine(out_channels, init_gamma=0.1, init_beta=-26) 

    def forward(self, x, ees):
        ees_amp, _ = ees.max(2)
        ees_spike = ees.clamp_(max=1)
        
        afferents = self.kinematic2afferent(x) # BxCxT
        
        ees_amp = ees_amp.repeat(1, afferents.size(1))          # BxC
        ees_recruitment = torch.sigmoid(self.affine(ees_amp)).unsqueeze(2)  # BxCx1
        ees_rate = F.relu(self.spike2rate(ees_spike))
        ees_ortho = ees_recruitment * ees_rate                             # BxCxT

        ees_anti = F.relu(self.antidromic(ees_ortho.view(-1,1,ees_ortho.shape[2])))
        ees_anti = ees_anti.view(*ees_ortho.shape)
        return ees_ortho + F.relu(afferents - ees_anti)