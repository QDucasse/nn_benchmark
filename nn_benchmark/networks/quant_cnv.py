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

import torch
from torch.nn import Module, ModuleList, BatchNorm2d, MaxPool2d, BatchNorm1d
from brevitas.nn import QuantConv2d, QuantHardTanh, QuantLinear
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.scaling import ScalingImplType

from nn_benchmark.networks.quant_utils import TensorNorm, make_quant_conv2d, make_quant_linear, make_quant_hard_tanh, get_quant_type



# QuantConv2d configuration
CNV_PADDING = 0
CNV_STRIDE  = 1
CNV_GROUPS  = 1
CNV_KERNEL  = 3
CNV_OUT_CH_POOL = [(64, False), (64, True), (128, False), (128, True), (256, False), (256, False)]

# Intermediate QuantLinear configuration
INTERMEDIATE_FC_PER_OUT_CH_SCALING = False
INTERMEDIATE_FC_FEATURES = [(256, 512), (512, 512)]

# Last QuantLinear configuration
LAST_FC_IN_FEATURES = 512
LAST_FC_PER_OUT_CH_SCALING = False

# MaxPool2d configuration
POOL_SIZE = 2

#
BIAS_ENABLED = False
WEIGHT_SCALING_IMPL_TYPE = ScalingImplType.CONST
WEIGHT_SCALING_CONST = 1.0

class QuantCNV(Module):

    def __init__(self, n_classes=10, weight_bit_width=1, act_bit_width=1, in_bit_width=8, in_channels=3):
        super(QuantCNV, self).__init__()

        weight_quant_type = get_quant_type(weight_bit_width)
        act_quant_type = get_quant_type(act_bit_width)
        in_quant_type = get_quant_type(in_bit_width)
        max_in_val = 1-2**(-7) # for Q1.7 input format
        self.conv_features = ModuleList()
        self.linear_features = ModuleList()

        self.conv_features.append(QuantHardTanh(bit_width=in_bit_width,
                                                quant_type=in_quant_type,
                                                max_val=max_in_val,
                                                restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
                                                scaling_impl_type=ScalingImplType.CONST))

        for out_ch, is_pool_enabled in CNV_OUT_CH_POOL:
            self.conv_features.append(make_quant_conv2d(in_channels=in_channels,
                                                        out_channels=out_ch,
                                                        bit_width=weight_bit_width,
                                                        kernel_size  = CNV_KERNEL,
                                                        stride       = CNV_STRIDE,
                                                        padding      = CNV_PADDING,
                                                        groups       = CNV_GROUPS,
                                                        bias         = False))
            in_ch = out_ch
            self.conv_features.append(BatchNorm2d(in_ch, eps=1e-4))
            self.conv_features.append(make_quant_hard_tanh(act_bit_width))
            if is_pool_enabled:
                self.conv_features.append(MaxPool2d(kernel_size=POOL_SIZE))

        for in_features, out_features in INTERMEDIATE_FC_FEATURES:
            self.linear_features.append(make_quant_linear(in_channels=in_features,
                                                          out_channels=out_features,
                                                          weight_scaling_per_output_channel=INTERMEDIATE_FC_PER_OUT_CH_SCALING,
                                                          bias=False,
                                                          bit_width=weight_bit_width))
            self.linear_features.append(BatchNorm1d(out_features, eps=1e-4))
            self.linear_features.append(make_quant_hard_tanh(act_bit_width))

        self.linear_features.append(make_quant_linear(in_channels=LAST_FC_IN_FEATURES,
                                                      out_channels=n_classes,
                                                      weight_scaling_per_output_channel=LAST_FC_PER_OUT_CH_SCALING,
                                                      bias=False,
                                                      bit_width=weight_bit_width))
        self.linear_features.append(TensorNorm())

        for m in self.modules():
          if isinstance(m, QuantConv2d) or isinstance(m, QuantLinear):
            torch.nn.init.uniform_(m.weight.data, -1, 1)

        self.name = "QuantCNV"

    def clip_weights(self, min_val, max_val):
        for mod in self.conv_features:
            if isinstance(mod, QuantConv2d):
                mod.weight.data.clamp_(min_val, max_val)
        for mod in self.linear_features:
            if isinstance(mod, QuantLinear):
                mod.weight.data.clamp_(min_val, max_val)

    def forward(self, x):
        x = 2.0 * x - torch.tensor([1.0], device=x.device)
        for mod in self.conv_features:
            x = mod(x)
        x = x.view(x.shape[0], -1)
        for mod in self.linear_features:
            x = mod(x)
        return x
