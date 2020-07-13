# -*- coding: utf-8 -*-

# nn_benchmark
# author - Quentin Ducasse
# https://github.com/QDucasse
# quentin.ducasse@ensta-bretagne.org

# Quantized AlexNet

import torch
import torch.nn as nn
import brevitas.nn as qnn

from nn_benchmark.networks.quant_utils import make_quant_conv2d, make_quant_linear, make_quant_avg_pool
from nn_benchmark.networks.quant_utils import make_quant_relu, make_quant_hard_tanh, make_quant_tanh, make_quant_sigmoid


class QuantAlexNet(nn.Module):
    '''AlexNet architecture in PyTorch'''
    def __init__(self, n_classes=10, in_channels=3,
                 weight_bit_width=4, act_bit_width=4, in_bit_width=8):
        super(QuantAlexNet, self).__init__()
        self.features = nn.Sequential(
            make_quant_conv2d(bit_width    = in_bit_width,
                              in_channels  = in_channels,
                              out_channels = 64,
                              kernel_size  = 11,
                              stride       = 4,
                              padding      = 2),
            make_quant_relu(bit_width = act_bit_width),
            nn.MaxPool2d(kernel_size=3, stride=1),
            make_quant_conv2d(bit_width    = weight_bit_width,
                              in_channels  = 64,
                              out_channels = 192,
                              kernel_size  = 5,
                              padding      = 2),
            make_quant_relu(bit_width = act_bit_width),
            nn.MaxPool2d(kernel_size=3, stride=1),
            make_quant_conv2d(bit_width    = weight_bit_width,
                              in_channels  = 192,
                              out_channels = 384,
                              kernel_size  = 3,
                              padding      = 1),
            make_quant_relu(bit_width = act_bit_width),
            make_quant_conv2d(bit_width    = weight_bit_width,
                              in_channels  = 384,
                              out_channels = 256,
                              kernel_size  = 3,
                              padding      = 1),
            make_quant_relu(bit_width = act_bit_width),
            make_quant_conv2d(bit_width    = weight_bit_width,
                              in_channels  = 256,
                              out_channels = 256,
                              kernel_size  = 3,
                              padding      = 1),
            make_quant_relu(bit_width = act_bit_width),
            nn.MaxPool2d(kernel_size=3, stride=1),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            make_quant_linear(bit_width    = weight_bit_width,
                              in_channels  = 256 * 6 * 6,
                              out_channels = 4096),
            make_quant_relu(bit_width = act_bit_width),
            nn.Dropout(),
            make_quant_linear(bit_width    = weight_bit_width,
                              in_channels  = 4096,
                              out_channels = 4096),
            make_quant_relu(bit_width = act_bit_width),
            make_quant_linear(bit_width = weight_bit_width,
                              in_channels = 4096,
                              out_channels = n_classes)
        )

        self.name = "QuantAlexNet"

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
