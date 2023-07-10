import torch
from .styled_vnet2 import StyledVNet

class NbodyD2DStyledVNet(StyledVNet):
    def __init__(self, style_size, in_chan, out_chan, bypass=None, **kwargs):
        """Nbody ZA (linear theory) displacements to Nbody nonlinear displacements
           V-Net like network with styles
           See `vnet.VNet`.
        """
        super(NbodyD2DStyledVNet, self).__init__(style_size, in_chan, out_chan, bypass, **kwargs)

    def forward(self, x, Om, Dz):

        s0 = (Om - 0.3) * 5.
        s1 = (Dz - 1.)

        s = torch.cat((s0.unsqueeze(0), s1.unsqueeze(0)), dim=1)

        x = x * Dz

        x = super().forward(x, s)

        return x, s
