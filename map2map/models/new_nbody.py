from new_styled_conv import StyleConv3d, StyleResNetBlock3d

class StyleVNet(nn.Module) :
    def __init__(self, style_chan, in_chan, out_chan=None):
        
        chan_1 = 18
        chan_2 = 64
        chan_3 = 128
        chan_4 = 82
        
        super().__init__()    
        self.conv_in = StyleResNetBlock3d('CACA', style_chan, 3, chan_1).to(device)
        self.conv_f1 = StyleResNetBlock3d('CACA', style_chan, chan_1, chan_1).to(device)
        self.down_1 = StyleResampleBlock3d('DA', style_chan, chan_1, chan_2).to(device)
        self.conv_f2 = StyleResNetBlock3d('CACA', style_chan, chan_2, chan_2).to(device)
        self.down_2 = StyleResampleBlock3d('DA', style_chan, chan_2, chan_3).to(device)
        self.conv_f3 = StyleResNetBlock3d('CACA', style_chan, chan_3, chan_2).to(device)
        self.down_3 = StyleResampleBlock3d('DA', style_chan, chan_2, chan_3).to(device)
        self.conv_c = StyleResNetBlock3d('CACA', style_chan, chan_3, chan_2).to(device)
        self.up_3 = StyleResampleBlock3d('UA', style_chan, chan_2, chan_2).to(device)
        self.conv_b3 = StyleResNetBlock3d('CACA', style_chan, chan_2 + chan_2, chan_3).to(device)
        self.up_2 = StyleResampleBlock3d('UA', style_chan, chan_3, chan_2).to(device)
        self.conv_b2 = StyleResNetBlock3d('CACA', style_chan, chan_2 + chan_2, chan_3).to(device)
        self.up_1 = StyleResampleBlock3d('UA', style_chan, chan_3, chan_2).to(device)
        self.conv_b1 = StyleResNetBlock3d('CACA', style_chan, chan_2 + chan_1, chan_4).to(device)
        self.conv_out = StyleResNetBlock3d('CACA', style_chan, chan_4, 3).to(device)
    
    def forward(self, x, s) :

        y = self.conv_in(x, s)

        x = narrow_by(x, 48)
        y = self.conv_f1(y, s)
        y1 = narrow_by(y, 40)
        y = self.down_1(y, s)

        y = self.conv_f2(y, s)
        y2 = narrow_by(y, 16)
        y = self.down_2(y, s)

        y = self.conv_f3(y, s)
        y3 = narrow_by(y, 4)

        y = self.down_3(y, s)

        y = self.conv_c(y, s)
        y = self.up_3(y, s)

        y = torch.cat([y3, y], axis = 1)
        del y3
        y = self.conv_b3(y, s)
        y = self.up_2(y, s)

        y = torch.cat([y2, y], axis = 1)
        del y2
        y = self.conv_b2(y, s)
        y = self.up_1(y, s)

        y = torch.cat([y1, y], axis = 1)
        del y1
        y = self.conv_b1(y, s)
        y = self.conv_out(y, s)

        y = y + x
        
        return y


class NbodyD2DStyledVNet(StyleVNet) :
    def __init__(self, style_size, in_chan, out_chan, **kwargs):
        """Nbody ZA (linear theory) displacements to Nbody nonlinear displacements
           V-Net like network with styles
           See `vnet.VNet`.
        """
        super(StyleVNet, self).__init__(style_size, in_chan, out_chan, **kwargs)


    def forward(self, x, Om, Dz):

        # Construct the style parameters
        s0 = Om - 0.3
        s1 = Dz - 1.
        s = torch.cat((s0.unsqueeze(0), s1.unsqueeze(0)), dim=1)

        # Rescale the ZA field
        x = x * Dz

        x = super().forward(x, s)

        return x, s
