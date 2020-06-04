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

from nn_benchmark.networks.quant_utils import make_quant_conv2d, make_quant_linear, make_quant_avg_pool
from nn_benchmark.networks.quant_utils import make_quant_relu, make_quant_hard_tanh, make_quant_tanh, make_quant_sigmoid


class DWConvPWConv(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1,
                 weight_bit_width=1, act_bit_width=1):
        super(DWConvPWConv, self).__init__()
        self.dw = nn.Sequential(
            make_quant_conv2d(in_channels=in_planes,
                              out_channels=in_planes,
                              kernel_size=3,
                              stride=stride,
                              padding=1,
                              groups=in_planes,
                              bias=False,
                              bit_width=weight_bit_width),
            nn.BatchNorm2d(in_planes),
            make_quant_relu(act_bit_width)
        )
        self.pw = nn.Sequential(
            make_quant_conv2d(in_channels=in_planes,
                              out_channels=out_planes,
                              kernel_size=1,
                              stride=1,
                              padding=0,
                              bias=False,
                              groups=1,
                              bit_width=weight_bit_width),
            nn.BatchNorm2d(out_planes),
            make_quant_relu(act_bit_width)
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
                 weight_bit_width=2, act_bit_width=2, in_bit_width=8):
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
            nn.BatchNorm2d(32),
            make_quant_relu(act_bit_width),
            self._make_layers(in_planes=32,
                              weight_bit_width=weight_bit_width,
                              act_bit_width=act_bit_width),
            nn.AvgPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024, n_classes)
        )
        self.name = "QuantMobilenetV1"

    def _make_layers(self, in_planes,weight_bit_width=1,act_bit_width=1):
        layers = []
        for tuple in self.cfg:
            out_planes, stride = tuple[0], tuple[1]
            layers.append(DWConvPWConv(in_planes=in_planes, out_planes=out_planes, stride=stride,
                                       weight_bit_width=weight_bit_width, act_bit_width=act_bit_width))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.feature_extractor(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return F.softmax(out,dim=1)
