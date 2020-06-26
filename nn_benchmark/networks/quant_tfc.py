# -*- coding: utf-8 -*-
# MIT License
#
# Copyright (c) 2019 Xilinx
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Included in:
# nn_benchmark
# author - Quentin Ducasse
# https://github.com/QDucasse
# quentin.ducasse@ensta-bretagne.org

# TFC network

from functools import reduce
from operator import mul

import torch
import torch.nn as nn
import brevitas.nn as bnn

from nn_benchmark.networks.quant_utils import make_quant_linear
from nn_benchmark.networks.quant_utils import make_quant_relu, make_quant_hard_tanh, make_quant_tanh, make_quant_sigmoid


FC_OUT_FEATURES = [64, 64, 64]
INTERMEDIATE_FC_PER_OUT_CH_SCALING = True
LAST_FC_PER_OUT_CH_SCALING = False
IN_DROPOUT = 0.2
HIDDEN_DROPOUT = 0.2


class QuantTFC(nn.Module):

    def __init__(self, n_classes=10, weight_bit_width=4, act_bit_width=4,
                 in_bit_width=8, in_channels=3, in_features=(32, 32)):
        super(QuantTFC, self).__init__()

        self.features = nn.ModuleList()
        self.features.append(make_quant_hard_tanh(in_bit_width))
        self.features.append(nn.Dropout(p=IN_DROPOUT))
        in_features = reduce(mul, in_features) * in_channels
        for out_features in FC_OUT_FEATURES:
            self.features.append(make_quant_linear(in_channels=in_features,
                                                   out_channels=out_features,
                                                   weight_scaling_per_output_channel=INTERMEDIATE_FC_PER_OUT_CH_SCALING,
                                                   bit_width=weight_bit_width))
            in_features = out_features
            self.features.append(nn.BatchNorm1d(num_features=in_features))
            self.features.append(make_quant_hard_tanh(act_bit_width))
            self.features.append(nn.Dropout(p=HIDDEN_DROPOUT))
        self.features.append(make_quant_linear(in_channels=in_features,
                                               out_channels=n_classes,
                                               weight_scaling_per_output_channel=LAST_FC_PER_OUT_CH_SCALING,
                                               bit_width=weight_bit_width))
        self.features.append(nn.BatchNorm1d(num_features=n_classes))

        self.name = "QuantTFC"

        # In case of binary activations and weights
        # for m in self.modules():
        #   if isinstance(m, bnn.QuantLinear):
        #     torch.nn.init.uniform_(m.weight.data, -1, 1)

    # In case of binary activations and weights
    # def clip_weights(self, min_val, max_val):
    #     for mod in self.features:
    #         if isinstance(mod, bnn.QuantLinear):
    #             mod.weight.data.clamp_(min_val, max_val)

    def forward(self, x):
        out = x.view(x.shape[0], -1)
        # In case of binary activations and weights
        # out = 2.0 * out - torch.tensor([1.0], device=out.device)
        for mod in self.features:
            out = mod(out)
        return out
