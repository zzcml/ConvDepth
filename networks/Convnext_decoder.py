from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from layers import *

class Resblock(nn.Module):
    def __init__(self,in_c,out_c):
        super(Resblock, self).__init__()
        self.conv3x3=Conv3x3(in_c,out_c)
        self.relu=nn.ReLU()
    def forward(self,x):
        # Input: [N, C, H, W], e.g., [1, 256, 32, 32]
        out=x  # [N, C, H, W]
        x=self.conv3x3(x)  # [N, out_c, H, W], Conv3x3 preserves spatial size
        x=self.relu(x)  # [N, out_c, H, W], ReLU activation
        x = self.conv3x3(x)  # [N, out_c, H, W]
        x = self.relu(x)  # [N, out_c, H, W]

        output=out+x  # [N, out_c, H, W], residual connection

        return output


class Enh(nn.Module):
    def __init__(self,in_c,out_c):
        super(Enh, self).__init__()
        self.conv1x1 = nn.Conv2d(out_c, out_c, 1)
        self.resblock=Resblock(out_c,out_c)
        self.convb = ConvBlock(in_c, out_c)

    def forward(self,input):
        # Input: tuple (x, y), e.g., x=[1, 256, 64, 64], y=[1, 256, 32, 32]
        # Note: x is upsampled feature from previous level, y is skip connection from encoder
        x,y=input
        bs, c, h, w = y.shape  # [N, C, H, W], e.g., [1, 256, 32, 32]
        Add=x + y  # [N, C, H, W], element-wise addition (x should be upsampled to match y)
        Att=self.resblock(y)  # [N, C, H, W], attention map from ResBlock
        Att=self.conv1x1(Att)  # [N, C, H, W], 1x1 conv for channel reduction
        Att=F.softmax(Att.view(bs,c,-1),dim=-1)  # [N, C, H*W], spatial softmax attention
        Mul=Att * Add.view(bs,c,-1)  # [N, C, H*W], attention-weighted features
        Mul=Mul.view(bs, c, h, w)  # [N, C, H, W], reshape back
        Add=Add.view(bs, c, h, w)  # [N, C, H, W]
        z=self.resblock(Mul)  # [N, C, H, W], feature refinement
        z=[z] + [Add]  # List of two tensors: [refined, residual]
        z=torch.cat(z,1)  # [N, 2*C, H, W], concatenate along channel dimension
        z = self.convb(z)  # [N, out_c, H, W], reduce channels back
        z=upsample(z)  # [N, out_c, 2H, 2W], upsample by factor of 2

        return z




class ConvDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(ConvDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_enc_0=np.array([256,256,512,1024])
        self.num_ch_dec = np.array([128, 128,256, 512])

        # decoder
        self.convs = OrderedDict()
        self.dec = nn.ModuleList()
        for i in range(4):
            dec=nn.Sequential(
                              *[Enh(self.num_ch_enc_0[i],self.num_ch_dec[i])]
                )
            self.dec.append(dec)

            self.convs[("conv1x1", i,0)]=ODConv2d(self.num_ch_enc[i], self.num_ch_enc[i], 1)
            self.convs[("conv3x3", i, 1)] = ConvBlock(self.num_ch_enc[i], self.num_ch_dec[i])


        for s in self.scales:
            self.convs[("dispconv", s)] = Head(self.num_ch_dec[s])

        self.decoder = nn.ModuleList(list(self.convs.values()))


    def forward(self, input_features):
        # Input: list of 4 encoder features from ConvNeXtV2
        # input_features[0]: [1, 128, 256, 256]
        # input_features[1]: [1, 256, 128, 128]
        # input_features[2]: [1, 512, 64, 64]
        # input_features[3]: [1, 1024, 32, 32]
        
        self.outputs = {}
        
        # Apply 1x1 conv to adjust encoder features
        f3 = self.convs[("conv1x1", 3,0)](input_features[3])  # [1, 1024, 32, 32] -> [1, 1024, 32, 32]
        f2 = self.convs[("conv1x1", 2,0)](input_features[2])  # [1, 512, 64, 64] -> [1, 512, 64, 64]
        f1 = self.convs[("conv1x1", 1,0)](input_features[1])  # [1, 256, 128, 128] -> [1, 256, 128, 128]
        f0 = self.convs[("conv1x1", 0,0)](input_features[0])  # [1, 128, 256, 256] -> [1, 128, 256, 256]

        # Decoder stage 3 (coarsest): process f3 and upsample
        x=self.convs[("conv3x3", 3,1)](f3)  # [1, 1024, 32, 32] -> [1, 512, 32, 32]
        x = upsample(x)  # [1, 512, 64, 64], upsample by 2x
        x=self.dec[3]((x,f2))  # Enh module: ([1, 512, 64, 64], [1, 512, 64, 64]) -> [1, 512, 128, 128]
        self.outputs[("disp", 3)]= self.convs[("dispconv", 3)](x)  # [1, 1, 128, 128], disparity output at scale 3

        # Decoder stage 2: process and upsample
        x=self.convs[("conv3x3", 2,1)](x)  # [1, 512, 128, 128] -> [1, 256, 128, 128]
        x=self.dec[2]((x,f1))  # Enh module: ([1, 256, 128, 128], [1, 256, 128, 128]) -> [1, 256, 256, 256]
        self.outputs[("disp", 2)]= self.convs[("dispconv", 2)](x)  # [1, 1, 256, 256], disparity output at scale 2

        # Decoder stage 1: process and upsample
        x=self.convs[("conv3x3", 1,1)](x)  # [1, 256, 256, 256] -> [1, 128, 256, 256]
        x=self.dec[1]((x,f0))  # Enh module: ([1, 128, 256, 256], [1, 128, 256, 256]) -> [1, 128, 512, 512]
        self.outputs[("disp", 1)] = self.convs[("dispconv", 1)](x)  # [1, 1, 512, 512], disparity output at scale 1

        # Decoder stage 0 (finest): process final level
        x=self.convs[("conv3x3", 0,1)](x)  # [1, 128, 512, 512] -> [1, 128, 512, 512]
        f0_up=upsample(f0)  # [1, 128, 512, 512], upsample skip connection
        x=self.dec[0]((x,f0_up))  # Enh module: ([1, 128, 512, 512], [1, 128, 512, 512]) -> [1, 128, 1024, 1024]
        self.outputs[("disp", 0)]= self.convs[("dispconv", 0)](x)  # [1, 1, 1024, 1024], final disparity output

        return self.outputs

