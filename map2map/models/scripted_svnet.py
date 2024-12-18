import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from .narrow import narrow_by
from typing import Optional, Tuple

class LeakyReLUStyled(nn.Module):
    def __init__(self, negative_slope: float = 0.01, inplace: bool = False):
        super(LeakyReLUStyled, self).__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, x: torch.Tensor, s: Optional[torch.Tensor] = None) -> torch.Tensor:
        return F.leaky_relu(x, negative_slope=self.negative_slope, inplace=self.inplace)

class StyleBase3d(nn.Module):
    def __init__(self, style_size: int, in_chan: int, out_chan: int, hidden_size: int, kernel_size: int, stride: int, spatial_in_shape: Tuple[int],
         spatial_out_shape: Tuple[int], transpose: bool, eps: float = 1e-8):
        super().__init__()

        self.style_size = style_size
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.hidden_size = hidden_size
        self.K3 = (kernel_size,) * 3
        self.stride = stride
        self.spatial_in_shape = spatial_in_shape
        self.spatial_out_shape = spatial_out_shape
        self.N = spatial_in_shape[0]
        self.Cin = spatial_in_shape[1]
        self.DHWin = spatial_in_shape[2:]
        self.DHWout = spatial_out_shape[2:]
        self.eps = eps

        #self.style_mlp = nn.Sequential(
        #    nn.Linear(self.style_size, self.hidden_size),
        #    nn.LeakyReLU(),
        #    nn.Linear(self.hidden_size, self.hidden_size),
        #    nn.LeakyReLU(),
        #    nn.Linear(self.hidden_size, self.in_chan)
        #)
        #for layer in self.style_mlp :
        #    if isinstance(layer, nn.Linear) :
        #        nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5), mode='fan_in', nonlinearity='leaky_relu')

        self.style_weight = nn.Parameter(torch.empty(in_chan, style_size))
        nn.init.kaiming_uniform_(self.style_weight, a=math.sqrt(5),
                                 mode='fan_in', nonlinearity='leaky_relu')
        self.style_bias = nn.Parameter(torch.ones(in_chan))
       
        if transpose :
            self.style_reshape = (self.N, self.Cin, 1, 1, 1, 1)
            self.fan_in_dim = (1, 3, 4, 5)
            self.conv = F.conv_transpose3d
            self.weight = nn.Parameter(torch.empty(self.in_chan, self.out_chan, *self.K3))
        else :
            self.style_reshape = (self.N, 1, self.Cin, 1, 1, 1)
            self.fan_in_dim = (2, 3, 4, 5)
            self.conv = F.conv3d
            self.weight = nn.Parameter(torch.empty(self.out_chan, self.in_chan, *self.K3))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5), mode='fan_in', nonlinearity='leaky_relu')

        self.bias = nn.Parameter(torch.zeros(out_chan))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

        self.w_reshape = (self.N * self.out_chan, self.in_chan, self.K3[0], self.K3[1], self.K3[2])
        self.x_reshape = (self.N, self.out_chan, self.DHWout[0], self.DHWout[1], self.DHWout[2]) 
            
    @torch.jit.export
    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor :
        
        #s = self.style_mlp(s)
        s = F.linear(s, self.style_weight, bias=self.style_bias)
        s = s.reshape(self.style_reshape)
        w = self.weight * s
        w = w * torch.rsqrt(w.pow(2).sum(dim=self.fan_in_dim, keepdim=True) + self.eps)
        
        w = w.reshape(self.w_reshape)
        x = x.reshape(1, self.N * self.Cin, self.DHWin[0], self.DHWin[1], self.DHWin[2])
        x = self.conv(x, w, bias=self.bias, stride=self.stride, groups=self.N)
        x = x.reshape(self.x_reshape)

        return x

class StyleConv3d(StyleBase3d):
    def __init__(self, style_size: int, in_chan: int, out_chan: int, hidden_size: int, spatial_in_shape: Tuple[int], eps: float = 1e-8):

        kernel_size = 3
        stride = 1
        spatial_out_shape = tuple(s - 2 if i > 1 else s for i, s in enumerate(spatial_in_shape))
        transpose = False

        super(StyleConv3d, self).__init__(style_size, in_chan, out_chan, hidden_size, kernel_size, stride, spatial_in_shape, spatial_out_shape, transpose, eps)
        
class StyleSkip3d(StyleBase3d):
    def __init__(self, style_size: int, in_chan: int, out_chan: int, hidden_size: int, spatial_in_shape: Tuple[int], eps: float = 1e-8):

        kernel_size = 1
        stride = 1
        spatial_out_shape = spatial_in_shape
        transpose = False

        super(StyleSkip3d, self).__init__(style_size, in_chan, out_chan, hidden_size, kernel_size, stride, spatial_in_shape, spatial_out_shape, transpose, eps)

class StyleDownSample3d(StyleBase3d):
    def __init__(self, style_size: int, in_chan: int, out_chan: int, hidden_size: int, spatial_in_shape: Tuple[int], eps: float = 1e-8):

        kernel_size = 2
        stride = 2
        spatial_out_shape = tuple(s // 2 if i > 1 else s for i, s in enumerate(spatial_in_shape))
        transpose = False

        super(StyleDownSample3d, self).__init__(style_size, in_chan, out_chan, hidden_size, kernel_size, stride, spatial_in_shape, spatial_out_shape, transpose, eps)
                    
class StyleUpSample3d(StyleBase3d):
    def __init__(self, style_size: int, in_chan: int, out_chan: int, hidden_size: int, spatial_in_shape: Tuple[int], eps: float = 1e-8):

        kernel_size = 2
        stride = 2
        spatial_out_shape = tuple(s * 2 if i > 1 else s for i, s in enumerate(spatial_in_shape))
        transpose = True

        super(StyleUpSample3d, self).__init__(style_size, in_chan, out_chan, hidden_size, kernel_size, stride, spatial_in_shape, spatial_out_shape, transpose, eps)

        
class StyleResampleBlock3d(nn.Module) :
    
    def __init__(self, seq: str, style_size: int, in_chan: int, out_chan: int, spatial_in_shape: Tuple[int]) :
        
        super().__init__()
        
        self.seq = seq
        self.style_size = style_size
        self.in_chan = in_chan
        self.out_chan = out_chan
        
        layer_spatial_in_shape = spatial_in_shape
        mid_chan = max(in_chan, out_chan)
        layers = []
        for idx_conv, l in enumerate(seq) :
            in_chan = out_chan = mid_chan
            if idx_conv == 0 :
                in_chan = self.in_chan
            if idx_conv == len(self.seq) - 1 :
                out_chan = self.out_chan
            layer_spatial_in_shape = tuple(in_chan if i == 1 else s for i, s in enumerate(layer_spatial_in_shape))
            if l == 'U':
                layer = StyleUpSample3d(self.style_size, in_chan, out_chan, 2 * self.style_size, layer_spatial_in_shape)
                layer_spatial_in_shape = layer.spatial_out_shape
            elif l == 'D':
                layer = StyleDownSample3d(self.style_size, in_chan, out_chan, 2 * self.style_size, layer_spatial_in_shape)
                layer_spatial_in_shape = layer.spatial_out_shape
            elif l == 'A':
                layer = LeakyReLUStyled()
            else :
                raise ValueError('layer type {} not supported'.format(l))
            layers.append(layer)
            
        self.convs = nn.ModuleList(layers)
        self.spatial_out_shape = layer_spatial_in_shape
              
    @torch.jit.export
    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor :
        for l in self.convs :
            x = l(x,s)
        return x

class StyleResNetBlock3d(nn.Module) :
    def __init__(self, seq: str, style_size: int, in_chan: int, out_chan: int, spatial_in_shape: Tuple[int], last_act: bool) :
        super().__init__()
                    
        self.style_size = style_size
        self.seq = seq
        self.in_chan = in_chan
        self.out_chan = out_chan
                
        layer_spatial_in_shape = spatial_in_shape
        mid_chan = max(in_chan, out_chan)
        layers = []
        for idx_conv, l in enumerate(seq) :
            in_chan = out_chan = mid_chan
            if idx_conv == 0 :
                in_chan = self.in_chan
            if idx_conv == len(self.seq) - 1 :
                out_chan = self.out_chan
            layer_spatial_in_shape = tuple(in_chan if i == 1 else s for i, s in enumerate(layer_spatial_in_shape))
            if l == 'C':
                layer = StyleConv3d(self.style_size, in_chan, out_chan, 2 * self.style_size, layer_spatial_in_shape)
                layer_spatial_in_shape = layer.spatial_out_shape
            elif l == 'A':
                layer = LeakyReLUStyled()
            else :
                raise ValueError('layer type {} not supported'.format(l))
            layers.append(layer)
            
        self.convs = nn.ModuleList(layers)
        self.spatial_out_shape = layer_spatial_in_shape
        self.skip = StyleSkip3d(style_size, self.in_chan, self.out_chan, 2 * self.style_size, spatial_in_shape)
        self.num_conv = sum([seq.count(l) for l in ['C']])
                
        if last_act :
            self.act = LeakyReLUStyled()
        else :
            self.act = None

    @torch.jit.export
    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor :
        y = self.skip(x, s)
        y = y[:,:,self.num_conv:-self.num_conv,self.num_conv:-self.num_conv,self.num_conv:-self.num_conv]
        for l in self.convs :
            x = l(x, s)
        x = x + y
        if self.act is not None :
            x = self.act(x)
        return x
