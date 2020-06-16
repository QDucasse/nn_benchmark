# -*- coding: utf-8 -*-

# nn_benchmark
# author - Quentin Ducasse
# https://github.com/QDucasse
# quentin.ducasse@ensta-bretagne.org

# Quantized Mobilenet architecture in PyTorch
# Taken from https://arxiv.org/abs/1704.04861

import torch
import torch.nn as nn
import torch.nn.functional as F
import brevitas.nn as qnn
from brevitas.quant_tensor import pack_quant_tensor

from nn_benchmark.networks.quant_utils import make_quant_conv2d, make_quant_linear, make_quant_avg_pool
from nn_benchmark.networks.quant_utils import make_quant_relu, make_quant_hard_tanh, make_quant_tanh, make_quant_sigmoid


class DWConvPWConv(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_channels, out_channels, stride=1,
                 weight_bit_width=4, act_bit_width=4,
                 activation_scaling_per_channel=False,bn_eps=1e-5):
        super(DWConvPWConv, self).__init__()
        self.dw = nn.Sequential(
            make_quant_conv2d(bit_width=weight_bit_width,
                              in_channels=in_channels,
                              out_channels=in_channels,
                              kernel_size=3,
                              stride=stride,
                              padding=1,
                              groups=in_channels,
                              bias=False),
            nn.BatchNorm2d(num_features=in_channels, eps=bn_eps),
            make_quant_relu(bit_width=act_bit_width,
                            per_channel_broadcastable_shape=(1, in_channels, 1, 1),
                            scaling_per_channel=activation_scaling_per_channel,
                            return_quant_tensor=True)
        )
        self.pw = nn.Sequential(
            make_quant_conv2d(bit_width=weight_bit_width,
                              in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=1,
                              stride=1,
                              padding=0,
                              bias=False,
                              groups=1),
            nn.BatchNorm2d(num_features=out_channels, eps=bn_eps),
            make_quant_relu(bit_width=act_bit_width,
                            per_channel_broadcastable_shape=(1, out_channels, 1, 1),
                            scaling_per_channel=activation_scaling_per_channel,
                            return_quant_tensor=True)
        )

    def forward(self, x):
        out = self.dw(x)
        out = self.pw(out)
        return out


class QuantMobilenetV1(nn.Module):
    '''Mobilenet architecture in PyTorch'''
    # (128,2) means conv planes=128, conv stride=2
    cfg = [(64,1),
           (128,2), (128,1),
           (256,2), (256,1),
           (512,2), (512,1), (512,1), (512,1), (512,1), (512,1),
           (1024,2), (1024,1)]

    def __init__(self, n_classes=10, in_channels=3,
                 weight_bit_width=4, act_bit_width=4, in_bit_width=8,
                 bn_eps=1e-5):
        super(QuantMobilenetV1, self).__init__()
        self.feature_extractor = nn.Sequential(
            make_quant_conv2d(in_channels=in_channels,
                              out_channels=32,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias=False,
                              groups=1,
                              bit_width=in_bit_width),
            nn.BatchNorm2d(num_features=32, eps=bn_eps),
            make_quant_relu(bit_width=act_bit_width,
                            scaling_per_channel=True,
                            per_channel_broadcastable_shape=(1, 32, 1, 1),
                            return_quant_tensor=True),
            self._make_layers(in_channels=32,
                              weight_bit_width=weight_bit_width,
                              act_bit_width=act_bit_width),
        )

        self.final_pool = make_quant_avg_pool(kernel_size=2,
                                              stride=1,
                                              signed=False,
                                              bit_width=weight_bit_width)

        self.classifier = make_quant_linear(in_channels=1024,
                                            out_channels=n_classes,
                                            bit_width=weight_bit_width)

        self.name = "QuantMobilenetV1"

    def _make_layers(self, in_channels,weight_bit_width=1,act_bit_width=1,bn_eps=1e-5):
        layers = []
        for i,tuple in enumerate(self.cfg):
            out_channels, stride = tuple[0], tuple[1]
            pw_activation_scaling_per_channel = i < out_channels - 1
            layers.append(DWConvPWConv(in_channels=in_channels, out_channels=out_channels, stride=stride,
                                       weight_bit_width=weight_bit_width, act_bit_width=act_bit_width,
                                       activation_scaling_per_channel=pw_activation_scaling_per_channel, bn_eps=bn_eps))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.feature_extractor(x)
        out = self.final_pool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return F.softmax(out, dim=1)
