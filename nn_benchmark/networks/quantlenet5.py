# -*- coding: utf-8 -*-

# nn_benchmark
# author - Quentin Ducasse
# https://github.com/QDucasse
# quentin.ducasse@ensta-bretagne.org

# Quantized LeNet5

import torch
import torch.nn as nn
import torch.nn.functional as F

from nn_benchmark.networks.util import make_quant_conv2d, make_quant_linear, make_quant_relu

# Conv2D parameters
CNV_PADDING = 0
CNV_STRIDE  = 1
CNV_GROUPS  = 1
CNV_KERNEL  = 5
CNV_OUT_CH  = [6, 16, 120]

# Dropout parameters
DROP_PROB = 0.2

# MaxPool2D parmeters
POOL_SIZE = 2

# FC parameters
FC_IN_FEAT = [120, 84]

class QuantLeNet5(nn.Module):
    '''LeNet neural network'''
    def __init__(self, num_classes=10, in_channels=1,
                 weight_bit_width=1, act_bit_width=2, in_bit_width=1):
        super(QuantLeNet5, self).__init__()

        self.feature_extractor = nn.Sequential(
            # First sequence: CONV => RELU => MAXPOOL
            make_quant_conv2d(in_channels  = in_channels,
                              out_channels = CNV_OUT_CH[0],
                              kernel_size  = CNV_KERNEL,
                              stride       = CNV_STRIDE,
                              padding      = CNV_PADDING,
                              groups       = CNV_GROUPS,
                              bias         = False,
                              bit_width    = in_bit_width),
            make_quant_relu(act_bit_width),
            nn.MaxPool2d(kernel_size=POOL_SIZE),
            # Second sequence: CONV => RELU => MAXPOOL
            make_quant_conv2d(in_channels  = CNV_OUT_CH[0],
                              out_channels = CNV_OUT_CH[1],
                              kernel_size  = CNV_KERNEL,
                              stride       = CNV_STRIDE,
                              padding      = CNV_PADDING,
                              groups       = CNV_GROUPS,
                              bias         = False,
                              bit_width    = weight_bit_width),
            make_quant_relu(act_bit_width),
            nn.MaxPool2d(kernel_size=POOL_SIZE),
            # Third sequence: CONV => RELU
            make_quant_conv2d(in_channels  = CNV_OUT_CH[1],
                              out_channels = CNV_OUT_CH[2],
                              kernel_size  = CNV_KERNEL,
                              stride       = CNV_STRIDE,
                              padding      = CNV_PADDING,
                              groups       = CNV_GROUPS,
                              bias         = False,
                              bit_width    = weight_bit_width),
            make_quant_relu(act_bit_width)
        )

        self.classifier = nn.Sequential(
            make_quant_linear(in_channels  = FC_IN_FEAT[0],
                              out_channels = FC_IN_FEAT[1],
                              bias         = False,
                              bit_width    = weight_bit_width),
            make_quant_relu(act_bit_width),
            make_quant_linear(in_channels  = FC_IN_FEAT[1],
                              out_channels = num_classes,
                              bias         = False,
                              bit_width    = weight_bit_width)
        )
        self.name = "QuantLeNet5"

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return probs
