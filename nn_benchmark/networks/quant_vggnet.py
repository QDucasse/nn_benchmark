# -*- coding: utf-8 -*-

# nn_benchmark
# author - Quentin Ducasse
# https://github.com/QDucasse
# quentin.ducasse@ensta-bretagne.org

# VGGNet 11/13/16/19 architecture in PyTorch
# Taken from https://arxiv.org/abs/1409.1556

import torch
import torch.nn as nn
import torch.nn.functional as F
import brevitas.nn as qnn

from nn_benchmark.networks.quant_utils import make_quant_conv2d, make_quant_linear, make_quant_avg_pool
from nn_benchmark.networks.quant_utils import make_quant_relu, make_quant_hard_tanh, make_quant_tanh, make_quant_sigmoid


cfg = {
    'QuantVGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'QuantVGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'QuantVGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'QuantVGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class QuantVGG(nn.Module):
    '''QuantVGG architectures in PyTorch'''
    def __init__(self, vgg_name, in_channels=3, n_classes=10,
                 weight_bit_width=2, act_bit_width=2, in_bit_width=8):
        super(QuantVGG, self).__init__()
        self.features = self._make_layers(cfg=cfg[vgg_name],
                                          in_channels=in_channels,
                                          weight_bit_width=weight_bit_width,
                                          act_bit_width=act_bit_width,
                                          in_bit_width=in_bit_width)
        self.classifier = make_quant_linear(in_channels=512, out_channels=n_classes, bit_width=weight_bit_width)
        self.name = vgg_name

    def _make_layers(self, cfg, in_channels,
                     weight_bit_width=2, act_bit_width=2, in_bit_width=8):
        '''Create the convolutions repetition based on a given configuration'''
        layers = []
        in_channels = in_channels
        # First step has to use the input bit width and is therefore extracted from the loop
        layers += [make_quant_conv2d(in_channels=in_channels,
                                     out_channels=64,
                                     kernel_size=3,
                                     bias=False,
                                     groups=1,
                                     stride=1,
                                     padding=1,
                                     bit_width=in_bit_width),
                   nn.BatchNorm2d(64),
                   make_quant_tanh(bit_width=act_bit_width)]
        # Others steps:
        # If the item of the configuration is an 'M' --> MaxPool layer
        # Else --> Conv(out=item) => BatchNorm => ReLU
        in_channels = 64
        for x in cfg[1:]:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                out_channels = x
                layers += [make_quant_conv2d(in_channels=in_channels,
                                             out_channels=out_channels,
                                             kernel_size=3,
                                             padding=1,
                                             bias=False,
                                             groups=1,
                                             stride=1,
                                             bit_width=weight_bit_width),
                           nn.BatchNorm2d(out_channels),
                           make_quant_tanh(bit_width=act_bit_width)]
                in_channels = out_channels
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return F.softmax(out,dim=1)


def QuantVGG11(in_channels=3, n_classes=10,weight_bit_width=2, act_bit_width=2, in_bit_width=8):
    '''QuantVGG11 configuration'''
    return QuantVGG("QuantVGG11",in_channels=in_channels,n_classes=n_classes,
                    weight_bit_width=weight_bit_width, act_bit_width=act_bit_width, in_bit_width=in_bit_width)

def QuantVGG13(in_channels=3, n_classes=10,weight_bit_width=2, act_bit_width=2, in_bit_width=8):
    '''QuantVGG13 configuration'''
    return QuantVGG("QuantVGG13",in_channels=in_channels,n_classes=n_classes,
                    weight_bit_width=weight_bit_width, act_bit_width=act_bit_width, in_bit_width=in_bit_width)

def QuantVGG16(in_channels=3, n_classes=10,weight_bit_width=2, act_bit_width=2, in_bit_width=8):
    '''QuantVGG16 configuration'''
    return QuantVGG("QuantVGG16",in_channels=in_channels,n_classes=n_classes,
                    weight_bit_width=weight_bit_width, act_bit_width=act_bit_width, in_bit_width=in_bit_width)

def QuantVGG19(in_channels=3, n_classes=10,weight_bit_width=2, act_bit_width=2, in_bit_width=8):
    '''QuantVGG19 configuration'''
    return QuantVGG("QuantVGG19",in_channels=in_channels,n_classes=n_classes,
                    weight_bit_width=weight_bit_width, act_bit_width=act_bit_width, in_bit_width=in_bit_width)
