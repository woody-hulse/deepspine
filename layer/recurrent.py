import torch
from torch import nn
import torch.nn.functional as F

import math
# class StackedLSTM(nn.Module):
#     # simple stacked lstm (no top-down pathway)
#     def __init__(self, input_size, hidden_sizes, output_size):
#         super(StackedLSTM, self).__init__()
#         self.output_size = output_size
#         self.cfg = hidden_sizes

#         lstms = []
#         for (i_size, h_size) in zip([input_size, *self.cfg[:-1]], self.cfg):
#             lstm = nn.LSTMCell(i_size, h_size)
#             lstms.append(lstm)

#         self.lstms = nn.ModuleList(lstms)
#         self.n_layers = len(lstms)
#         self.output_linear = nn.Linear(self.cfg[-1], output_size)

#     def _init_zero_state(self, batch_size, device):
#         hiddens = []
#         cells = []

#         for h_size in self.cfg:
#             hiddens.append(torch.zeros(batch_size, h_size).to(device))
#             cells.append(torch.zeros(batch_size, h_size).to(device))

#         return hiddens, cells

#     def forward(self, xs):
#         batch_size = xs.size(1)
#         device = xs.device

#         hiddens, cells = self._init_zero_state(batch_size, device)
#         ys = []
#         last_hiddens = []

#         for x in xs:
#             for lstm in self.lstms:
#                 hx = hiddens[-self.n_layers]
#                 cx = cells[-self.n_layers]

#                 h, c = lstm(x, (hx, cx))
                
#                 hiddens.append(h)
#                 cells.append(c)

#                 x = h

#             last_hiddens.append(h)

#         last_hiddens = torch.stack(last_hiddens, dim=0)
#         _y = self.output_linear(last_hiddens.view(-1, last_hiddens.size(2)))
#         y = (_y.view(-1, batch_size, self.output_size))

#         return y

# class StackedGRU(nn.Module):
#     # simple stacked gru (no top-down pathway)
#     def __init__(self, input_size, hidden_sizes, Tmax):
#         super(StackedGRU, self).__init__()
#         self.cfg = hidden_sizes
#         self.is_emb = is_emb

#         grus = []
        
#         for (i_size, h_size) in zip([input_size, *self.cfg[:-1]], self.cfg):
#             gru = nn.GRUCell(i_size, h_size)
#             gru.bias_ih.data.fill_(0)
#             gru.bias_hh.data.fill_(0)
            
#             torch.nn.init.uniform_(gru.bias_hh[h_size:2*h_size].data, 1, Tmax - 1)
#             gru.bias_hh[h_size:2*h_size].data.log_().mul_(-1)
#             print(gru.bias_hh[h_size:2*h_size])
#             grus.append(gru)

#         self.grus = nn.ModuleList(grus)
#         self.n_layers = len(grus)

#     def _init_zero_state(self, batch_size, device):
#         hiddens = []

#         for h_size in self.cfg:
#             hiddens.append(torch.zeros(batch_size, h_size).to(device))

#         return hiddens

#     def forward(self, xs):
#         batch_size = xs.size(1)
#         device = xs.device

#         hiddens = self._init_zero_state(batch_size, device)
#         ys = []
#         last_hiddens = []

#         for x in xs:
#             if self.is_emb:
#                 x = F.relu(self.embedding(x))
#                 # x = F.tanh(self.embedding(x))

#             for gru in self.grus:
#                 hx = hiddens[-self.n_layers]

#                 h = gru(x, hx)
                
#                 # hiddens.append(hx)

#                 hiddens.append(h)
#                 x = h

#             last_hiddens.append(h)

#         last_hiddens = torch.stack(last_hiddens, dim=0)

#         return last_hiddens

class StackedGRU(nn.Module):
    # simple stacked gru (no top-down pathway)
    def __init__(self, input_size, hidden_size, Tmax, dropout=None, output_norm=None):
        super(StackedGRU, self).__init__()
        if isinstance(hidden_size, int):
            hidden_size = [hidden_size]

        self.hidden_size = hidden_size

        grus = []
        
        for hidden_size in self.hidden_size:
            gru = nn.GRUCell(input_size, hidden_size)
            
            self._chrono_init(gru, hidden_size, Tmax)
            grus.append(gru)

            input_size = hidden_size

        self.grus = nn.ModuleList(grus)
        self.n_layers = len(grus)

        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

        if output_norm is not None:
            self.output_norm = nn.InstanceNorm1d(input_size, affine=True)
            nn.init.constant_(self.output_norm.weight, 0.1)
            nn.init.constant_(self.output_norm.bias, 0)
        else:
            self.output_norm = None

    def _chrono_init(self, gru, hidden_size, Tmax):
        gru.bias_ih.data.fill_(0)
        gru.bias_hh.data.fill_(0)
        
        torch.nn.init.uniform_(gru.bias_hh[hidden_size:2*hidden_size].data, 1, Tmax - 1)
        gru.bias_hh[hidden_size:2*hidden_size].data.log_()#.mul_(-1)
            
    def _init_zero_state(self, batch_size, device):
        hiddens = []

        for hidden_size in self.hidden_size:
            hiddens.append(torch.zeros(batch_size, hidden_size).to(device))

        return hiddens

    def forward(self, xs):
        # xs: [batch x dim x time]
        batch_size = xs.size(0)
        T = xs.size(2)
        device = xs.device

        hiddens = self._init_zero_state(batch_size, device)
        ys = []
        last_hiddens = []

        if self.dropout is not None:
            xs = self.dropout(xs)

        for t in range(T):
            x = xs[:,:,t]
            for gru in self.grus:
                hx = hiddens[-self.n_layers]

                h = gru(x, hx)
                
                hiddens.append(h)
                x = h

            last_hiddens.append(h)

        last_hiddens = torch.stack(last_hiddens, dim=2)
        return last_hiddens



class LayerNormGRUCell(nn.Module):

    """
    An implementation of GRUCell.

    """

    def __init__(self, input_size, hidden_size, Tmax):
        super(LayerNormGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Tmax = Tmax
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=False)
        self.x2h_norm = nn.LayerNorm(3 * hidden_size)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.h2h_norm = nn.LayerNorm(3 * hidden_size)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

        nn.init.constant_(self.x2h_norm.weight, 0.1)
        nn.init.constant_(self.x2h_norm.bias, 0)
        nn.init.constant_(self.h2h_norm.weight, 0.1)
        nn.init.constant_(self.h2h_norm.bias, 0)

        torch.nn.init.uniform_(self.h2h_norm.bias[self.hidden_size:2*self.hidden_size].data, 1, self.Tmax - 1)
        self.h2h_norm.bias[self.hidden_size:2*self.hidden_size].data.log_()#.mul_(-1)
            

    def forward(self, x, hidden):
        # import pdb
        # pdb.set_trace()
        # x = x.view(-1, x.size(1))
        
        gate_x = self.x2h_norm(self.x2h(x))
        gate_h = self.h2h_norm(self.h2h(hidden))
        
        # gate_x = gate_x.squeeze()
        # gate_h = gate_h.squeeze()
        
        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)
        
        
        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        newgate = F.tanh(i_n + (resetgate * h_n))
        
        hy = newgate + inputgate * (hidden - newgate)
        
        
        return hy

class StackedLayerNormGRU(nn.Module):
    # simple stacked gru (no top-down pathway)
    def __init__(self, input_size, hidden_size, Tmax, offset=0, dropout=None):
        super(StackedLayerNormGRU, self).__init__()
        if isinstance(hidden_size, int):
            hidden_size = [hidden_size]

        self.hidden_size = hidden_size
        self.offset = offset

        grus = []
        
        for hidden_size in self.hidden_size:
            gru = LayerNormGRUCell(input_size, hidden_size, Tmax)
            grus.append(gru)

            input_size = hidden_size

        self.grus = nn.ModuleList(grus)
        self.n_layers = len(grus)

        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

        # if output_norm is not None:
        #     self.output_norm = nn.InstanceNorm1d(input_size, affine=True)
        #     nn.init.constant_(self.output_norm.weight, 0.1)
        #     nn.init.constant_(self.output_norm.bias, 0)
        # else:
        #     self.output_norm = None

            
    def _init_zero_state(self, batch_size, device):
        hiddens = []

        for hidden_size in self.hidden_size:
            hiddens.append(torch.zeros(batch_size, hidden_size).to(device))

        return hiddens

    def forward(self, xs, ees=None):
        # xs: [batch x dim x time]
        batch_size = xs.size(0)
        T = xs.size(2)
        device = xs.device

        hiddens = self._init_zero_state(batch_size, device)
        ys = []
        last_hiddens = []

        if self.dropout is not None:
            xs = self.dropout(xs)

        if ees is not None:
            xs = torch.cat([xs, ees], dim=1)

        for t in range(T):
            x = xs[:,:,t]
            for gru in self.grus:
                hx = hiddens[-self.n_layers]

                h = gru(x, hx)
                
                hiddens.append(h)
                x = h

            last_hiddens.append(h)

        last_hiddens = torch.stack(last_hiddens, dim=2)
        if self.offset > 0:
            return last_hiddens[:,:,self.offset:]
        else:
            return last_hiddens