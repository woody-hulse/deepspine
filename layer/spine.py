import torch
from torch import nn
import torch.nn.functional as F

from .utils import LayerNorm, Affine, LinearLN

class Integrator(nn.Module):
    def __init__(self, out_neurons, Tmax, activation):
        super(Integrator, self).__init__()
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'softplus':
            self.activation = nn.Softplus()
        else:
            self.activation = None

        self.Tmax = Tmax
        self.x2h = nn.Sequential(  
            nn.Linear(out_neurons, out_neurons, bias=False),
            nn.LayerNorm(out_neurons)
        )

        self.h2h = nn.Sequential(  
            nn.Linear(out_neurons, 2*out_neurons, bias=False),
            nn.LayerNorm(2*out_neurons)
        )

        self._reset_parameters()
        
    def _reset_parameters(self):
        # chrono initialization
        nn.init.constant_(self.x2h[-1].weight, 0.1)
        self.x2h[-1].bias.data = -torch.log(torch.nn.init.uniform_(self.x2h[-1].bias.data, 1, self.Tmax - 1))
        
        nn.init.constant_(self.h2h[-1].weight, 0.1)
        nn.init.constant_(self.h2h[-1].bias, 0)

    def forward(self, x, hx):
        batch_size = x.shape[0]

        h_i, h_g = self.h2h(hx).chunk(2,dim=1)
        
        x = self.activation(x + h_i)

        # compute a dynamic stepsize (Eq. (4))
        _g = self.x2h(x) + h_g
        g = torch.sigmoid(_g)

        h = (1 - g) * hx + g * x    # (Eq. (7))

        return h


class Layer(nn.Module):
    def __init__(self, 
            in_pathways, 
            out_neurons, 
            Tmax,
            activation,
            reciprocal_inhibition=False):
        super(Layer, self).__init__()

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'softplus':
            self.activation = nn.Softplus()
        else:
            self.activation = None
        
        self.norm = LayerNorm(dim=2, affine_shape=[2*in_pathways,out_neurons,1])

        self.reciprocal_inhibition = reciprocal_inhibition
        if self.reciprocal_inhibition:
            self.div_inhibition = LinearLN(2*out_neurons, out_neurons)

        self.in_pathways = in_pathways
        self.out_neurons = out_neurons

        self.affine_flx = Affine([in_pathways,out_neurons], init_gamma=0.1, init_beta=1)
        self.affine_ext = Affine([in_pathways,out_neurons], init_gamma=0.1, init_beta=1)

        self.flexor = Integrator(out_neurons, Tmax, activation)
        self.extensor = Integrator(out_neurons, Tmax, activation)

    def forward(self, xs, init_h, EI=None):
        # B: batch size
        # P: # of input pathways either for flexor and extensor
        # N: # of output neurons
        # T: # of time steps
        
        hiddens = [init_h]

        xs = torch.stack(xs, dim=1)         # [Bx(2xP)xNxT]
        xs = self.norm(xs) # apply layer normalization separately per pathway

        batch_size, T = xs.shape[0], xs.shape[-1]  

        x_flxs, x_exts = xs.chunk(2, dim=1) # x_flxs: [BxPxNxT], x_exts: [BxPxNxT]
        for t in range(T):
            hx = hiddens[-1]        # [Bx(2xN)]            
            hx_flx, hx_ext = hx.chunk(2, dim=1)
            x_flx = x_flxs[...,t]   # [BxPxN]
            x_ext = x_exts[...,t]   # [BxPxN]
            
            # compute a dynamic gain control (Eq. (6))
            g_flx = self.affine_flx(hx_flx.unsqueeze(1).repeat(1,self.in_pathways,1))
            g_ext = self.affine_ext(hx_ext.unsqueeze(1).repeat(1,self.in_pathways,1))
            
            g_flx = torch.sigmoid(g_flx)
            g_ext = torch.sigmoid(g_ext)

            in_flx = (g_flx * x_flx)
            in_ext = (g_ext * x_ext)

            if EI is not None:
                # EI: [P]
                # EI set to True for excitation and set to False for inhibition pathways
                in_flx = self.activation(in_flx[:,EI].sum(1)) - self.activation(in_flx[:,~EI].sum(1)) # [BxN]
                in_ext = self.activation(in_ext[:,EI].sum(1)) - self.activation(in_ext[:,~EI].sum(1)) # [BxN]
            else:
                in_flx = self.activation(in_flx.sum(1)) # [BxN]
                in_ext = self.activation(in_ext.sum(1)) # [BxN]

            if self.reciprocal_inhibition:
                # reciprocal inhibition betwen Iai flexors and extensors
                gate = torch.sigmoid(self.div_inhibition(hx))   # [BxN]
                in_flx = in_flx * gate
                in_ext = in_ext * (1 - gate)

            h_flx = self.flexor(in_flx, hx_flx)   # [BxN]
            h_ext = self.extensor(in_ext, hx_ext)   # [BxN]
            
            h = torch.cat([h_flx, h_ext], dim=1)    # [Bx(2xN)]        
            hiddens.append(h)

        return torch.stack(hiddens[1:], dim=2)    # exclude initial hidden state


class SpinalCordCircuit(nn.Module):
    def __init__(self, 
            Ia_neurons, 
            II_neurons, 
            ex_neurons, 
            Iai_neurons, 
            mn_neurons,
            Tmax,
            offset=0,
            activation='relu',
            dropout=None):
        super(SpinalCordCircuit, self).__init__()
        self.Ia_neurons = Ia_neurons
        self.II_neurons = II_neurons
        self.ex_neurons = ex_neurons
        self.Iai_neurons = Iai_neurons
        self.mn_neurons = mn_neurons

        self.offset = offset

        self.ex = Layer(
            in_pathways=1, 
            out_neurons=ex_neurons, 
            Tmax=Tmax,
            activation=activation
        )
        self.Iai = Layer(
            in_pathways=2, 
            out_neurons=Iai_neurons, 
            Tmax=Tmax,
            activation=activation,
            reciprocal_inhibition=True
        )
        self.mn = Layer(
            in_pathways=3, 
            out_neurons=mn_neurons, 
            Tmax=Tmax,
            activation=activation
        )

        conv1d = nn.Conv1d
        self.Ia2mnIai_connection = conv1d(2*Ia_neurons, 2*(Iai_neurons + mn_neurons), kernel_size=1, groups=2)
        self.II2exIai_connection = conv1d(2*II_neurons, 2*(Iai_neurons + ex_neurons), kernel_size=1, groups=2)
        self.ex2mn_connection = conv1d(2*ex_neurons, 2*mn_neurons, kernel_size=1, groups=2)
        self.Iai2mn_connection = conv1d(2*Iai_neurons, 2*mn_neurons, kernel_size=1, groups=2)

        # excitatory pathways (True): Ia2mn, ex2mn || inhibitory pathways (False): Iai2mn
        self.register_buffer('mn_connectivity', torch.BoolTensor([True, True, False])) 

        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

    def _init_hidden(self, x):
        batch_size = x.size(0)

        init_ex = x.new(batch_size, 2*self.ex_neurons).zero_()
        init_Iai = x.new(batch_size, 2*self.Iai_neurons).zero_()
        init_mn = x.new(batch_size, 2*self.mn_neurons).zero_()

        return init_ex, init_Iai, init_mn

    def forward(self, afferents, verbose=False):
        init_ex, init_Iai, init_mn = self._init_hidden(afferents)

        if self.dropout is not None:
            afferents = self.dropout(afferents)
        
        # Ia: [BxIaNxT], II: [BxIINxT] (IaN: # of Ia neurons, IIN: # of II neurons)
        Ia, II = torch.split(afferents, [2*self.Ia_neurons, 2*self.II_neurons], dim=1)  
        
        # compute inputs of Ex, Iai, Mn from afferents (Ia & II)
        Ia2mnIai = self.Ia2mnIai_connection(Ia)
        II2exIai = self.II2exIai_connection(II)

        ## split into flexor and extensor
        Ia2mnIai_flxs, Ia2mnIai_exts = Ia2mnIai.chunk(2, dim=1)
        II2exIai_flxs, II2exIai_exts = II2exIai.chunk(2, dim=1)

        Ia2mn_flxs, Ia2Iai_flxs = Ia2mnIai_flxs.split([self.mn_neurons, self.Iai_neurons], dim=1)
        II2ex_flxs, II2Iai_flxs = II2exIai_flxs.split([self.ex_neurons, self.Iai_neurons], dim=1)
        Ia2mn_exts, Ia2Iai_exts = Ia2mnIai_exts.split([self.mn_neurons, self.Iai_neurons], dim=1)
        II2ex_exts, II2Iai_exts = II2exIai_exts.split([self.ex_neurons, self.Iai_neurons], dim=1)

        # compute excitatory neurons
        exs = self.ex(
            [II2ex_flxs, II2ex_exts],
            init_ex
        )
        ex2mn = self.ex2mn_connection(exs)
        ex2mn_flxs, ex2mn_exts = ex2mn.chunk(2, dim=1)
        

        # compute inhibitory neurons
        Iais = self.Iai(
            [Ia2Iai_flxs, II2Iai_flxs, Ia2Iai_exts, II2Iai_exts],
            init_Iai
        )
        Iai2mn = self.Iai2mn_connection(Iais)
        Iai2mn_exts, Iai2mn_flxs = Iai2mn.chunk(2, dim=1) # swap the order of flexor and extensor for lateral inhibition

        
        # compute motor neurons
        mns = self.mn(
            [Ia2mn_flxs, ex2mn_flxs, Iai2mn_flxs, Ia2mn_exts, ex2mn_exts, Iai2mn_exts], 
            init_mn,
            self.mn_connectivity
        )
        if verbose:
            if self.offset > 0:
                return mns[:,:,self.offset:], Iais[:,:,self.offset:], exs[:,:,self.offset:]
            else:
                return mns, Iais, exs
        else:        
            if self.offset > 0:
                return mns[:,:,self.offset:]
            else:
                return mns #, hiddens