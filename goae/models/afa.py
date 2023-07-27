import math
import torch
import torch.nn as nn
from torch.nn import Conv2d, BatchNorm2d, PReLU, Sequential
import torch.nn.functional as F
from collections import namedtuple
from models.helpers import bottleneck_IR_SE
from models.attention import CrossAttention

"""
Borrowed from implementation from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch) 
and [StyleCLIP](https://github.com/orpatashnik/StyleCLIP)  with modifications
"""

class Bottleneck_F(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    """ A named tuple describing a ResNet block. """

def get_block_F(in_channel, depth, num_units, stride=2):
    return [Bottleneck_F(in_channel, depth, stride)] + [Bottleneck_F(depth, depth, 1) for i in range(num_units - 1)]

def get_blocks_F():
    blocks = [
        get_block_F(in_channel=64, depth=64, num_units=3),
        get_block_F(in_channel=64, depth=128, num_units=3),
        get_block_F(in_channel=128, depth=256, num_units=3),
    ]

    return blocks


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * math.sqrt(2)


class EqualConv2d(nn.Module):
    def __init__(
            self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )


        
class AFA(nn.Module):
    def __init__(self,  **unused):
        super(AFA, self).__init__()

        blocks = get_blocks_F()
        unit_module = bottleneck_IR_SE

        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.output_layer = nn.Sequential(
                    EqualConv2d(256, 512, 3, stride=1, padding=1, bias=True ),
                    ScaledLeakyReLU(0.2),
                    EqualConv2d(512, 512, 3, stride=1, padding=1, bias=True ))


        self.norm_kv = nn.LayerNorm(512, elementwise_affine=False)
        self.norm_x = nn.LayerNorm(512, elementwise_affine=False)

        ## treat the Linear projection in CrossAttention as 1*1 Convolution
        ## TODO: Release the code and checkpoint for high resolution F space AFA Module. 
        self.cross_att_gamma  = CrossAttention(512, 4, 1024, batch_first=True)          
        self.cross_att_beta  = CrossAttention(512, 4, 1024, batch_first=True)


    def forward(self, img, rec_img, feature_map):
        B,L,M,_ = feature_map.shape

        x =  rec_img - img
        x = self.input_layer(x)

        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)

        x = self.output_layer(x)

        kv = x.flatten(2).permute(0,2,1)
        kv = self.norm_kv(kv)

        feature_map_in = feature_map.flatten(2).permute(0,2,1)
        feature_map_in = self.norm_x(feature_map_in)

        feature_map_gamma = self.cross_att_gamma(feature_map_in, kv)
        feature_map_gamma = feature_map_gamma.permute(0,2,1).reshape(B,L,M,M)   

        feature_map_beta = self.cross_att_beta(feature_map_in, kv)
        feature_map_beta = feature_map_beta.permute(0,2,1).reshape(B,L,M,M)          

        feature_map_adain = feature_map_gamma * feature_map + feature_map_beta

        return feature_map_adain, feature_map_gamma, feature_map_beta