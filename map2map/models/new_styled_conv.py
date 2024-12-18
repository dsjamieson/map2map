import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from .narrow import narrow_by
from typing import Optional

class LeakyReLUStyled(nn.Module):
    def __init__(self, negative_slope: float = 0.01, inplace: bool = False):
        super(LeakyReLUStyled, self).__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, x: torch.Tensor, s: Optional[torch.Tensor] = None) -> torch.Tensor:
        return F.leaky_relu(x, negative_slope=self.negative_slope, inplace=self.inplace)


class StyleConv3d(nn.Module):
    def __init__(self, style_size, in_chan, out_chan, kernel_size=3, stride=1, bias=True, hidden_size = None, resample=None):
        super().__init__()

        if hidden_size is None :
            hidden_size = 2 * in_chan
        self.style_mlp = nn.Sequential(
            nn.Linear(style_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, in_chan)
        )
        for layer in self.style_mlp :
            if isinstance(layer, nn.Linear) :
                nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5), mode='fan_in', nonlinearity='leaky_relu')
        
        self.in_chan = in_chan
        self.out_chan = out_chan
        if resample == 'U':
            self.conv = F.conv_transpose3d
            self.in_out = [0,1]
            self.fan_in_dim = (1, 3, 4, 5)
            self.K3 = (2,) * 3
            self.stride = 2
            self.weight = nn.Parameter(torch.empty(self.in_chan, self.out_chan, *self.K3))
        else :
            self.conv = F.conv3d
            self.in_out = [1,0]
            self.fan_in_dim = (2, 3, 4, 5)
            if resample is None:
                self.K3 = (kernel_size,) * 3
                self.stride = stride
            elif resample == 'D':
                self.K3 = (2,) * 3
                self.stride = 2
            else:
                raise ValueError('resample type {} not supported'.format(resample))
            self.weight = nn.Parameter(torch.empty(self.out_chan, self.in_chan, *self.K3))

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5), mode='fan_in', nonlinearity='leaky_relu')

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_chan))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x, s, eps: float = 1e-8):
        
        N = x.shape[0]
        Cin = x.shape[1]
        DHWin = x.shape[2:] 
        s_in_shape = [Cin, 1][self.in_out[0]]
        s_out_shape = [Cin, 1][self.in_out[1]]

        s = self.style_mlp(s)
        s = s.reshape(N, s_in_shape, s_out_shape, 1, 1, 1)

        w = self.weight * s
        w = w * torch.rsqrt(w.pow(2).sum(dim=self.fan_in_dim, keepdim=True) + eps)

        w = w.reshape(N * self.out_chan, self.in_chan, self.K3[0], self.K3[1], self.K3[2])
        x = x.reshape(1, N * Cin, DHWin[0], DHWin[1], DHWin[2])
        x = self.conv(x, w, bias=self.bias, stride=self.stride, groups=N)

        DHWout = x.shape[2:] 
        x = x.reshape(N, self.out_chan, DHWout[0], DHWout[1], DHWout[2])

        return x

class StyleResampleBlock3d(nn.Module) :
    
    def __init__(self, seq, style_size, in_chan, out_chan=None, mid_chan=None, kernel_size=2, stride=2):
        super().__init__()
        self.act = LeakyReLUStyled()
        if out_chan is None:
            out_chan = in_chan
        self.seq = seq
        self.style_size = style_size
        self.in_chan = in_chan
        self.out_chan = out_chan
        if mid_chan is None:
            self.mid_chan = max(in_chan, out_chan)
        self.kernel_size = kernel_size
        self.stride = stride
        
        self.idx_conv = 0
        layers = [self._get_layer(l) for l in seq]
        self.convs = nn.ModuleList(layers)
        
    def _get_layer(self, l):
        in_chan = out_chan = self.mid_chan
        if self.idx_conv == 0 :
            in_chan = self.in_chan
        if self.idx_conv == len(self.seq) - 1 :
            out_chan = self.out_chan
        self.idx_conv += 1
        if l == 'U':
            return StyleConv3d(self.style_size, in_chan, out_chan, kernel_size=2, stride=2, resample = 'U')
        elif l == 'D':
            return StyleConv3d(self.style_size, in_chan, out_chan, kernel_size=2, stride=2, resample = 'D')
        elif l == 'A':
            return self.act
        else:
            raise ValueError('layer type {} not supported'.format(l))
            
    def forward(self, x, s):
        for l in self.convs :
            x = l(x,s)
        return x

class StyleResNetBlock3d(nn.Module) :
    def __init__(self, seq, style_size, in_chan, out_chan=None, mid_chan=None, kernel_size=3, stride=1, last_act = None):
        super().__init__()
        self.act = LeakyReLUStyled()

        if out_chan is None:
            out_chan = in_chan
            self.skip = None
        else:
            self.skip = StyleConv3d(style_size, in_chan, out_chan, 1)

        if last_act is None:
            last_act = seq[-1] == 'A'
        elif last_act and seq[-1] != 'A':
            warnings.warn(
                'Disabling last_act without trailing activation in seq',
                RuntimeWarning,
            )
            last_act = False
        if last_act:
            seq = seq[:-1]
            self.act = LeakyReLUStyled()
        else:
            self.act = None
            
        self.style_size = style_size
        self.seq = seq
        self.in_chan = in_chan
        self.out_chan = out_chan
        if mid_chan is None:
            self.mid_chan = max(in_chan, out_chan)
        self.kernel_size = kernel_size
        self.stride = stride
            
        self.idx_conv = 0
        self.num_conv = sum([seq.count(l) for l in ['C']])
        layers = [self._get_layer(l) for l in self.seq]
        self.convs = nn.ModuleList(layers)
        self.num_ops = len(layers)
        
    def _get_layer(self, l):
        in_chan = out_chan = self.mid_chan
        if self.idx_conv == 0 :
            in_chan = self.in_chan
        if self.idx_conv == len(self.seq) - 1 :
            out_chan = self.out_chan
        self.idx_conv += 1
        if l == 'C':
            return StyleConv3d(self.style_size, in_chan, out_chan, self.kernel_size, stride=self.stride)
        elif l == 'A':
            return LeakyReLUStyled()
        else:
            raise ValueError('layer type {} not supported'.format(l))

    def forward(self, x, s):
        y = x
        if self.skip is not None:
            y = self.skip(y, s)
            y = narrow_by(y, self.num_conv)
        for l in self.convs:
            x = l(x, s)
        if self.skip is not None:
            x += y
        if self.act is not None:
            x = self.act(x)
        return x
