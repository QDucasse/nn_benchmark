# -*- coding: utf-8 -*-

# nn_benchmark
# author - Quentin Ducasse
# https://github.com/QDucasse
# quentin.ducasse@ensta-bretagne.org

# Quantized LeNet5

import torch
import torch.nn as nn
import brevitas.nn as qnn

from nn_benchmark.networks.quant_utils import make_quant_conv2d, make_quant_linear, make_quant_avg_pool
from nn_benchmark.networks.quant_utils import make_quant_relu, make_quant_hard_tanh, make_quant_tanh, make_quant_sigmoid

# Conv2D parameters
CNV_PADDING = 0
CNV_STRIDE  = 1
CNV_GROUPS  = 1
CNV_KERNEL  = 5
CNV_OUT_CH  = [6, 16, 120]

# MaxPool2D parmeters
POOL_STRIDE = 2
POOL_SIZE   = 2

# FC parameters
FC_IN_FEAT = [120, 84]

class QuantLeNet5(nn.Module):
    '''LeNet5 neural network'''
    def __init__(self, n_classes=10, in_channels=3,
                 weight_bit_width=4, act_bit_width=4, in_bit_width=8):
        super(QuantLeNet5, self).__init__()

        self.feature_extractor = nn.Sequential(
            # First sequence: CONV => RELU => AVGPOOL
            make_quant_conv2d(bit_width    = in_bit_width,
                              in_channels  = in_channels,
                              out_channels = CNV_OUT_CH[0],
                              kernel_size  = CNV_KERNEL,
                              stride       = CNV_STRIDE,
                              padding      = CNV_PADDING,
                              groups       = CNV_GROUPS),
            make_quant_tanh(bit_width=act_bit_width,
                            return_quant_tensor=True),
            make_quant_avg_pool(bit_width   = max(2,weight_bit_width),
                                kernel_size = POOL_SIZE,
                                stride      = POOL_STRIDE),
            # nn.AvgPool2d(kernel_size=POOL_SIZE,stride=POOL_STRIDE),
            # Second sequence: CONV => RELU => AVGPOOL
            make_quant_conv2d(bit_width    = weight_bit_width,
                              in_channels  = CNV_OUT_CH[0],
                              out_channels = CNV_OUT_CH[1],
                              kernel_size  = CNV_KERNEL,
                              stride       = CNV_STRIDE,
                              padding      = CNV_PADDING,
                              groups       = CNV_GROUPS),
            make_quant_tanh(bit_width=act_bit_width,
                            return_quant_tensor=True),
            make_quant_avg_pool(bit_width   = max(2,weight_bit_width),
                                kernel_size = POOL_SIZE,
                                stride      = POOL_STRIDE),
            # nn.AvgPool2d(kernel_size=POOL_SIZE,stride=POOL_STRIDE),
            # Third sequence: CONV => RELU
            make_quant_conv2d(bit_width    = weight_bit_width,
                              in_channels  = CNV_OUT_CH[1],
                              out_channels = CNV_OUT_CH[2],
                              kernel_size  = CNV_KERNEL,
                              stride       = CNV_STRIDE,
                              padding      = CNV_PADDING,
                              groups       = CNV_GROUPS),
            make_quant_tanh(bit_width=act_bit_width)
        )

        self.classifier = nn.Sequential(
            make_quant_linear(bit_width    = weight_bit_width,
                              in_channels  = FC_IN_FEAT[0],
                              out_channels = FC_IN_FEAT[1]),
            make_quant_tanh(act_bit_width),
            make_quant_linear(bit_width    = weight_bit_width,
                              in_channels  = FC_IN_FEAT[1],
                              out_channels = n_classes)
        )

        # Use if Binary
        # self.initialize_weights()
        self.name = "QuantLeNet5"

    # Use if Binary and Bipolar
    # def initialize_weights(self):
    #     for m in self.modules():
    #       if isinstance(m, qnn.QuantConv2d) or isinstance(m, qnn.QuantLinear):
    #         torch.nn.init.uniform_(m.weight.data, -1, 1)

    def forward(self, x):
        out = self.feature_extractor(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
