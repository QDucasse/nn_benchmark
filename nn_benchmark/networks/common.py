# -*- coding: utf-8 -*-

# nn_benchmark
# author - Quentin Ducasse
# https://github.com/QDucasse
# quentin.ducasse@ensta-bretagne.org

import torch.nn as nn

def equal_conv(conv1,conv2):

    return ((conv1.in_channels  == conv2.in_channels)  and
            (conv1.out_channels == conv2.out_channels) and
            (conv1.kernel_size  == conv2.kernel_size)  and
            (conv1.stride       == conv2.stride)       and
            (conv1.padding      == conv2.padding)      and
            (conv1.dilation     == conv2.dilation)     and
            (conv1.groups       == conv2.groups)       and
            (conv1.padding_mode == conv2.padding_mode))

def equal_avg_pool(avgpool1,avgpool2):
        return ((avgpool1.kernel_size       == avgpool2.kernel_size)       and
                (avgpool1.stride            == avgpool2.stride)            and
                (avgpool1.padding           == avgpool2.padding)           and
                (avgpool1.ceil_mode         == avgpool2.ceil_mode)         and
                (avgpool1.count_include_pad == avgpool2.count_include_pad) and
                (avgpool1.divisor_override  == avgpool2.divisor_override))

def equal_linear(linear1, linear2):
    return ((linear1.in_features  == linear2.in_features) and
            (linear1.out_features == linear2.out_features))

def equal_relu(relu1, relu2):
    return relu1.inplace == relu2.inplace

def equal_hardtanh(hardtanh1, hardtanh2):
    return ((hardtanh1.min_val   == hardtanh2.min_val)   and
            (hardtanh1.max_val   == hardtanh2.max_val)   and
            (hardtanh1.inplace   == hardtanh2.inplace)   and
            (hardtanh1.min_value == hardtanh2.min_value) and
            (hardtanh1.max_value == hardtanh2.max_value))
