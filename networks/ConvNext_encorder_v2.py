# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import DropPath if available, otherwise use Identity as fallback
try:
    from timm.models.layers import trunc_normal_, DropPath
except ImportError:
    # Fallback when timm is not installed
    trunc_normal_ = None
    class DropPath(nn.Module):
        """Drop paths (Stochastic Depth) per sample - identity fallback"""
        def __init__(self, drop_prob=0.):
            super().__init__()
            self.drop_prob = drop_prob
        def forward(self, x):
            return x  # Identity during inference
        
# LayerNorm and GRN imports removed - using BatchNorm for hardware friendliness


class Block(nn.Module):
    """ ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv, [N,C,H,W]->[N,C,H,W]
        # Replace LayerNorm+permute with Conv2d-based normalization for hardware friendliness
        self.norm = nn.BatchNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)  # pointwise conv: [N,C,H,W]->[N,4C,H,W]
        self.act = nn.ReLU(inplace=True)  # ReLU: hardware friendly activation f(x)=max(0,x)
        # GRN removed for hardware efficiency - avoid global reduction operations
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)  # pointwise conv: [N,4C,H,W]->[N,C,H,W]
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # Input: [N, C, H, W] = [1, 3, 1024, 1024]
        input = x
        x = self.dwconv(x)  # [1, C, H, W], depthwise conv with kernel 7, padding 3
        x = self.norm(x)  # [1, C, H, W], BatchNorm (can be fused to conv for deployment)
        x = self.pwconv1(x)  # [1, 4*C, H, W], pointwise conv: C -> 4*C
        x = self.act(x)  # [1, 4*C, H, W], ReLU activation
        # GRN removed - skip global response normalization
        x = self.pwconv2(x)  # [1, C, H, W], pointwise conv: 4*C -> C

        x = input + self.drop_path(x)  # [1, C, H, W], residual connection
        return x


class ConvNeXtV2(nn.Module):
    """ ConvNeXt V2

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=3, num_classes=1000,
                 depths: list=[3, 3, 9, 3], dims: list=[96, 192, 384, 768],
                 drop_path_rate=0., head_init_scale=1.
                 ):
        super().__init__()
        self.num_ch_enc = [128, 256, 512, 1024]
        self.depths = depths
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            nn.BatchNorm2d(dims[0], eps=1e-6)
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.BatchNorm2d(dims[i], eps=1e-6),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        # self.head = nn.Linear(dims[-1], num_classes)
        #
        # self.apply(self._init_weights)
        # self.head.weight.data.mul_(head_init_scale)
        # self.head.bias.data.mul_(head_init_scale)

    # def _init_weights(self, m):
    #     if isinstance(m, (nn.Conv2d, nn.Linear)):
    #         trunc_normal_(m.weight, std=.02)
    #         nn.init.constant_(m.bias, 0)

    # def forward_features(self, x):
    #     for i in range(4):
    #         x = self.downsample_layers[i](x)
    #         x = self.stages[i](x)
    #     return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)
    #
    # def forward(self, x):
    #     x = self.forward_features(x)
    #     x = self.head(x)
    #     return x

    def layer(self,i,x):
        x=self.downsample_layers[i](x)
        x=self.stages[i](x)
        return x

    def forward(self, x):
        # Input: [N, C, H, W] = [1, 3, 1024, 1024]
        # Stage 0: stem downsampling (kernel=4, stride=4)
        self.features = []
        x = self.downsample_layers[0](x)  # [1, dims[0], 256, 256], e.g., [1, 128, 256, 256] for base model
        x = self.stages[0](x)  # [1, dims[0], 256, 256], after Block processing
        self.features.append(x)  # features[0]: [1, 128, 256, 256]
        
        # Stage 1: downsampling (stride=2)
        self.features.append(self.layer(1, self.features[-1]))  # [1, dims[1], 128, 128], e.g., [1, 256, 128, 128]
        
        # Stage 2: downsampling (stride=2)
        self.features.append(self.layer(2, self.features[-1]))  # [1, dims[2], 64, 64], e.g., [1, 512, 64, 64]
        
        # Stage 3: downsampling (stride=2)
        self.features.append(self.layer(3, self.features[-1]))  # [1, dims[3], 32, 32], e.g., [1, 1024, 32, 32]
        
        return self.features  # List of 4 feature maps at different scales



model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
    "convnext_v2_base_1k":"https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_base_1k_224_fcmae.pt"
}

def convnextv2_atto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model


def convnextv2_femto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model


def convnext_pico(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model


def convnextv2_nano(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model


def convnextv2_tiny(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model


def convnextv2_base(pretrained=True, in_22k=True,**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        url = model_urls['convnext_v2_base_1k'] if in_22k else model_urls['convnext_v2_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        print("loading pretrained model")
    return model


def convnextv2_large(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model


def convnextv2_huge(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model