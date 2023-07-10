import torch
import torch.nn as nn

from .styled_conv import ConvStyledBlock, ResStyledBlock
from .narrow import narrow_by


class StyledVNet(nn.Module):
    def __init__(self, style_size, in_chan, out_chan, bypass=None, **kwargs):
        """V-Net like network with styles

        See `vnet.VNet`.
        """
        super().__init__()

        # activate non-identity skip connection in residual block
        # by explicitly setting out_chan
        self.conv_l00 = ResStyledBlock(style_size, in_chan, 64, seq='CACA')
        self.conv_l01 = ResStyledBlock(style_size, 64, 64, seq='CACA')
        self.down_l0 = ConvStyledBlock(style_size, 64, seq='DA')
        self.conv_l1 = ResStyledBlock(style_size, 64, 64, seq='CACA')
        self.down_l1 = ConvStyledBlock(style_size, 64, seq='DA')
        self.conv_l2 = ResStyledBlock(style_size, 64, 64, seq='CACA')
        self.down_l2 = ConvStyledBlock(style_size, 64, seq='DA')

        self.conv_c = ResStyledBlock(style_size, 64, 64, seq='CACA')

        self.up_r2 = ConvStyledBlock(style_size, 64, seq='UA')
        self.conv_r2 = ResStyledBlock(style_size, 128, 64, seq='CACA')
        self.up_r1 = ConvStyledBlock(style_size, 64, seq='UA')
        self.conv_r1 = ResStyledBlock(style_size, 128, 64, seq='CACA')
        self.up_r0 = ConvStyledBlock(style_size, 64, seq='UA')
        self.conv_r00 = ResStyledBlock(style_size, 128, 64, seq='CACA')
        self.conv_r01 = ResStyledBlock(style_size, 64, out_chan, seq='CAC')

        if bypass is None:
            self.bypass = in_chan == out_chan
        else:
            self.bypass = bypass

    def forward(self, x, s):

        #print("forward : %.2e GB" % (torch.cuda.memory_allocated(0)/1024/1024/1024))
        if self.bypass:
            x0 = narrow_by(x, 48)
        #print("bypass  : %.2e GB" % (torch.cuda.memory_allocated(0)/1024/1024/1024))

        x = self.conv_l00(x, s)
        #print("conv_l00: %.2e GB" % (torch.cuda.memory_allocated(0)/1024/1024/1024))

        y0 = self.conv_l01(x, s)
        #print("conv_l01: %.2e GB" % (torch.cuda.memory_allocated(0)/1024/1024/1024))

        x = self.down_l0(y0, s)
        #print("down_l0 : %.2e GB" % (torch.cuda.memory_allocated(0)/1024/1024/1024))

        y0 = narrow_by(y0, 40)
        #print("narrow_0: %.2e GB" % (torch.cuda.memory_allocated(0)/1024/1024/1024))

        y1 = self.conv_l1(x, s)
        #print("conv_l1 : %.2e GB" % (torch.cuda.memory_allocated(0)/1024/1024/1024))

        x = self.down_l1(y1, s)
        #print("down_l1 : %.2e GB" % (torch.cuda.memory_allocated(0)/1024/1024/1024))

        y1 = narrow_by(y1, 16)
        #print("narrow_1: %.2e GB" % (torch.cuda.memory_allocated(0)/1024/1024/1024))

        y2 = self.conv_l2(x, s)
        #print("conv_l1 : %.2e GB" % (torch.cuda.memory_allocated(0)/1024/1024/1024))

        x = self.down_l2(y2, s)
        #print("down_l2 : %.2e GB" % (torch.cuda.memory_allocated(0)/1024/1024/1024))

        y2 = narrow_by(y2, 4)
        #print("narrow_2: %.2e GB" % (torch.cuda.memory_allocated(0)/1024/1024/1024))

        x = self.conv_c(x, s)
        #print("conv_c  : %.2e GB" % (torch.cuda.memory_allocated(0)/1024/1024/1024))

        x = self.up_r2(x, s)
        #print("up_r2   : %.2e GB" % (torch.cuda.memory_allocated(0)/1024/1024/1024))

        x = torch.cat([y2, x], dim=1)
        #print("cat_2   : %.2e GB" % (torch.cuda.memory_allocated(0)/1024/1024/1024))

        del y2
        torch.cuda.empty_cache()
        #print("del_2   : %.2e GB" % (torch.cuda.memory_allocated(0)/1024/1024/1024))

        x = self.conv_r2(x, s)
        #print("conv_r2 : %.2e GB" % (torch.cuda.memory_allocated(0)/1024/1024/1024))

        x = self.up_r1(x, s)
        #print("up_r1   : %.2e GB" % (torch.cuda.memory_allocated(0)/1024/1024/1024))

        x = torch.cat([y1, x], dim=1)
        #print("cat_1   : %.2e GB" % (torch.cuda.memory_allocated(0)/1024/1024/1024))

        del y1
        torch.cuda.empty_cache()
        #print("del_1   : %.2e GB" % (torch.cuda.memory_allocated(0)/1024/1024/1024))

        x = self.conv_r1(x, s)
        #print("conv_r2 : %.2e GB" % (torch.cuda.memory_allocated(0)/1024/1024/1024))

        x = self.up_r0(x, s)
        #print("up_r0   : %.2e GB" % (torch.cuda.memory_allocated(0)/1024/1024/1024))

        x = torch.cat([y0, x], dim=1)
        #print("cat_0   : %.2e GB" % (torch.cuda.memory_allocated(0)/1024/1024/1024))

        del y0
        torch.cuda.empty_cache()
        #print("del_0   : %.2e GB" % (torch.cuda.memory_allocated(0)/1024/1024/1024))

        x = self.conv_r00(x, s)
        #print("conv_r00: %.2e GB" % (torch.cuda.memory_allocated(0)/1024/1024/1024))

        x = self.conv_r01(x, s)
        #print("conv_r01: %.2e GB" % (torch.cuda.memory_allocated(0)/1024/1024/1024))

        if self.bypass:
            x += x0
            del x0
            torch.cuda.empty_cache()

        #print("done    : %.2e GB" % (torch.cuda.memory_allocated(0)/1024/1024/1024))

        return x
