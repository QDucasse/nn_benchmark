# -*- coding: utf-8 -*-

# nn_benchmark
# author - Quentin Ducasse
# https://github.com/QDucasse
# quentin.ducasse@ensta-bretagne.org

# Quantized layers helpers

from brevitas.core.quant import QuantType

from nn_benchmark.networks.common import equal_conv, equal_avg_pool, equal_linear

def get_quant_type(bit_width):
    '''Given a bitwidth, output the corresponding quantized type:
        - None   -> Floating point
        - 1      -> Binary
        - Others -> Integer'''
    if bit_width is None:
        return QuantType.FP
    elif bit_width == 1:
        return QuantType.BINARY
    else:
        return QuantType.INT


def test_args(args, elt1, elt2):
    res = True
    for arg in args:
        res = res and (elt1.init_args[arg] == elt2.init_args[arg])
    return res


def equal_quant_conv(conv1,conv2):
    args = ["weight_bit_width", "weight_scaling_impl_type",
            "weight_scaling_stats_op", "weight_scaling_per_output_channel",
            "weight_restrict_scaling_type", "weight_narrow_range",
            "weight_scaling_min_val", "bias", "bias_quant_type",
            "compute_output_bit_width", "compute_output_scale"]
    res = test_args(args,conv1,conv2) and equal_conv(conv1,conv2)
    return res

def equal_quant_avg_pool(avgpool1,avgpool2):
    return ((avgpool1.quant_type == avgpool2.quant_type) and
            (avgpool1.signed     == avgpool2.signed) and
            equal_avg_pool(avgpool1,avgpool2))

def equal_quant_linear(linear1,linear2):
    args = ["weight_bit_width", "weight_scaling_impl_type",
            "weight_scaling_stats_op", "weight_scaling_per_output_channel",
            "weight_restrict_scaling_type", "weight_narrow_range",
            "weight_scaling_min_val", "bias", "bias_quant_type",
            "compute_output_bit_width", "compute_output_scale"]
    res = test_args(args,linear1,linear2) and equal_linear(linear1,linear2)
    return res

def equal_quant_relu(relu1, relu2):
    args = ["bit_width", "scaling_impl_type",
            "scaling_per_channel", "restrict_scaling_type",
            "scaling_min_val", "max_val", "min_overall_bit_width",
            "max_overall_bit_width", "return_quant_tensor",
            "per_channel_broadcastable_shape"]
    return test_args(args,relu1,relu2)


def equal_quant_hardtanh(hardtanh1, hardtanh2):
    args = ["bit_width", "scaling_impl_type",
            "scaling_per_channel", "restrict_scaling_type",
            "scaling_min_val", "threshold", "min_overall_bit_width",
            "max_overall_bit_width", "return_quant_tensor",
            "per_channel_broadcastable_shape"]
    res = test_args(args,hardtanh1,hardtanh2)
    return res

def equal_quant_tanh(tanh1, tanh2):
    args = ["bit_width", "restrict_scaling_type", "return_quant_tensor"]
    return test_args(args,tanh1,tanh2)

def equal_quant_sigmoid(sigmoid1, sigmoid2):
    args = ["bit_width", "restrict_scaling_type", "return_quant_tensor"]
    return test_args(args,sigmoid1,sigmoid2)
