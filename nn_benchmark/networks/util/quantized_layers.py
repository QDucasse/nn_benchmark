# -*- coding: utf-8 -*-

# nn_benchmark
# author - Quentin Ducasse
# https://github.com/QDucasse
# quentin.ducasse@ensta-bretagne.org

# Quantized layers helpers

import brevitas.nn as qnn
from brevitas.core.quant        import QuantType
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.scaling      import ScalingImplType
from brevitas.core.stats        import StatsOp

# Global quantization
SCALING_MIN_VAL   = 2e-16
ENABLE_BIAS_QUANT = False

# Activation function quantization
ACT_SCALING_IMPL_TYPE               = ScalingImplType.PARAMETER
ACT_SCALING_PER_CHANNEL             = False
ACT_SCALING_RESTRICT_SCALING_TYPE   = RestrictValueType.LOG_FP
ACT_MAX_VAL                         = 6.0
ACT_RETURN_QUANT_TENSOR             = False
ACT_PER_CHANNEL_BROADCASTABLE_SHAPE = None
HARD_TANH_THRESHOLD                 = 10.0

# Weight quantization
WEIGHT_SCALING_IMPL_TYPE          = ScalingImplType.STATS
WEIGHT_SCALING_PER_OUTPUT_CHANNEL = True
WEIGHT_SCALING_STATS_OP           = StatsOp.MAX
WEIGHT_RESTRICT_SCALING_TYPE      = RestrictValueType.LOG_FP
WEIGHT_NARROW_RANGE               = True

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

def make_quant_conv2d(in_channels,
                      out_channels,
                      kernel_size,
                      stride,
                      padding,
                      groups,
                      bias,
                      bit_width,
                      enable_bias_quant=ENABLE_BIAS_QUANT,
                      weight_scaling_impl_type=WEIGHT_SCALING_IMPL_TYPE,
                      weight_scaling_stats_op=WEIGHT_SCALING_STATS_OP,
                      weight_scaling_per_output_channel=WEIGHT_SCALING_PER_OUTPUT_CHANNEL,
                      weight_restrict_scaling_type=WEIGHT_RESTRICT_SCALING_TYPE,
                      weight_narrow_range=WEIGHT_NARROW_RANGE,
                      weight_scaling_min_val=SCALING_MIN_VAL):
    '''Helper for Conv2D layers'''
    weight_quant_type = get_quant_type(bit_width)
    bias_quant_type = weight_quant_type if enable_bias_quant else QuantType.FP
    return qnn.QuantConv2d(in_channels,
                           out_channels,
                           groups=groups,
                           kernel_size=kernel_size,
                           padding=padding,
                           stride=stride,
                           bias=bias,
                           bias_quant_type=bias_quant_type,
                           compute_output_bit_width=bias and enable_bias_quant,
                           compute_output_scale=bias and enable_bias_quant,
                           weight_bit_width=bit_width,
                           weight_quant_type=weight_quant_type,
                           weight_scaling_impl_type=weight_scaling_impl_type,
                           weight_scaling_stats_op=weight_scaling_stats_op,
                           weight_scaling_per_output_channel=weight_scaling_per_output_channel,
                           weight_restrict_scaling_type=weight_restrict_scaling_type,
                           weight_narrow_range=weight_narrow_range,
                           weight_scaling_min_val=weight_scaling_min_val)


def make_quant_linear(in_channels,
                      out_channels,
                      bias,
                      bit_width,
                      enable_bias_quant=ENABLE_BIAS_QUANT,
                      weight_scaling_impl_type=WEIGHT_SCALING_IMPL_TYPE,
                      weight_scaling_stats_op=WEIGHT_SCALING_STATS_OP,
                      weight_scaling_per_output_channel=WEIGHT_SCALING_PER_OUTPUT_CHANNEL,
                      weight_restrict_scaling_type=WEIGHT_RESTRICT_SCALING_TYPE,
                      weight_narrow_range=WEIGHT_NARROW_RANGE,
                      weight_scaling_min_val=SCALING_MIN_VAL):
    '''Helper for Linear (Fully Connected) layers'''
    weight_quant_type = get_quant_type(bit_width)
    bias_quant_type = weight_quant_type if enable_bias_quant else QuantType.FP
    return qnn.QuantLinear(in_channels, out_channels,
                           bias=bias,
                           bias_quant_type=bias_quant_type,
                           compute_output_bit_width=bias and enable_bias_quant,
                           compute_output_scale=bias and enable_bias_quant,
                           weight_bit_width=bit_width,
                           weight_quant_type=weight_quant_type,
                           weight_scaling_impl_type=weight_scaling_impl_type,
                           weight_scaling_stats_op=weight_scaling_stats_op,
                           weight_scaling_per_output_channel=weight_scaling_per_output_channel,
                           weight_restrict_scaling_type=weight_restrict_scaling_type,
                           weight_narrow_range=weight_narrow_range,
                           weight_scaling_min_val=weight_scaling_min_val)


def make_quant_relu(bit_width,
                    scaling_impl_type=ACT_SCALING_IMPL_TYPE,
                    scaling_per_channel=ACT_SCALING_PER_CHANNEL,
                    restrict_scaling_type=ACT_SCALING_RESTRICT_SCALING_TYPE,
                    scaling_min_val=SCALING_MIN_VAL,
                    max_val=ACT_MAX_VAL,
                    return_quant_tensor=ACT_RETURN_QUANT_TENSOR,
                    per_channel_broadcastable_shape=ACT_PER_CHANNEL_BROADCASTABLE_SHAPE):
    '''Helper for ReLU activation layers'''
    quant_type = get_quant_type(bit_width)
    return qnn.QuantReLU(bit_width=bit_width,
                         quant_type=quant_type,
                         scaling_impl_type=scaling_impl_type,
                         scaling_per_channel=scaling_per_channel,
                         restrict_scaling_type=restrict_scaling_type,
                         scaling_min_val=scaling_min_val,
                         max_val=max_val,
                         return_quant_tensor=return_quant_tensor,
                         per_channel_broadcastable_shape=per_channel_broadcastable_shape)


def make_quant_hard_tanh(bit_width,
                         scaling_impl_type=ACT_SCALING_IMPL_TYPE,
                         scaling_per_channel=ACT_SCALING_PER_CHANNEL,
                         restrict_scaling_type=ACT_SCALING_RESTRICT_SCALING_TYPE,
                         scaling_min_val=SCALING_MIN_VAL,
                         threshold=HARD_TANH_THRESHOLD,
                         return_quant_tensor=ACT_RETURN_QUANT_TENSOR,
                         per_channel_broadcastable_shape=ACT_PER_CHANNEL_BROADCASTABLE_SHAPE):
    '''Helper for HardTanH activation layers'''
    quant_type = get_quant_type(bit_width)
    return qnn.QuantHardTanh(bit_width=bit_width,
                             quant_type=quant_type,
                             scaling_per_channel=scaling_per_channel,
                             scaling_impl_type=scaling_impl_type,
                             restrict_scaling_type=restrict_scaling_type,
                             scaling_min_val=scaling_min_val,
                             max_val=threshold,
                             min_val=-threshold,
                             per_channel_broadcastable_shape=per_channel_broadcastable_shape,
                             return_quant_tensor=return_quant_tensor)


def make_quant_avg_pool(bit_width,
                        kernel_size,
                        stride,
                        signed):
    '''Helper for AveragePooling layers'''
    quant_type = get_quant_type(bit_width)
    return qnn.QuantAvgPool2d(kernel_size=kernel_size,
                              quant_type=quant_type,
                              signed=signed,
                              stride=stride,
                              min_overall_bit_width=bit_width,
                              max_overall_bit_width=bit_width)
