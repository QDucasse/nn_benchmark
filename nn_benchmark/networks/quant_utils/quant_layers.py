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

from nn_benchmark.networks.quant_utils import get_quant_type

# Global quantization
SCALING_MIN_VAL   = 2e-16
ENABLE_BIAS       = False
ENABLE_BIAS_QUANT = False

# Weight quantization
WEIGHT_SCALING_IMPL_TYPE          = ScalingImplType.STATS
WEIGHT_SCALING_PER_OUTPUT_CHANNEL = True
WEIGHT_SCALING_STATS_OP           = StatsOp.MAX
WEIGHT_RESTRICT_SCALING_TYPE      = RestrictValueType.LOG_FP
WEIGHT_NARROW_RANGE               = True
WEIGHT_RETURN_QUANT_TENSOR        = False

# Conv2d Defaults
CNV_KERNEL_SIZE = 3
CNV_STRIDE      = 1
CNV_PADDING     = 0
CNV_GROUPS      = 1

# AvgPool2d Defaults
POOL_KERNEL_SIZE = 2
POOL_STRIDE      = 2
POOL_SIGNED      = False

def make_quant_conv2d(bit_width,
                      in_channels,
                      out_channels,
                      kernel_size=CNV_KERNEL_SIZE,
                      stride=CNV_STRIDE,
                      padding=CNV_PADDING,
                      groups=CNV_GROUPS,
                      bias=ENABLE_BIAS,
                      enable_bias_quant=ENABLE_BIAS_QUANT,
                      weight_scaling_impl_type=WEIGHT_SCALING_IMPL_TYPE,
                      weight_scaling_stats_op=WEIGHT_SCALING_STATS_OP,
                      weight_scaling_per_output_channel=WEIGHT_SCALING_PER_OUTPUT_CHANNEL,
                      weight_restrict_scaling_type=WEIGHT_RESTRICT_SCALING_TYPE,
                      weight_narrow_range=WEIGHT_NARROW_RANGE,
                      weight_scaling_min_val=SCALING_MIN_VAL,
                      return_quant_tensor=WEIGHT_RETURN_QUANT_TENSOR):
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
                           weight_scaling_min_val=weight_scaling_min_val,
                           return_quant_tensor=return_quant_tensor)


def make_quant_linear(bit_width,
                      in_channels,
                      out_channels,
                      bias=ENABLE_BIAS,
                      enable_bias_quant=ENABLE_BIAS_QUANT,
                      weight_scaling_impl_type=WEIGHT_SCALING_IMPL_TYPE,
                      weight_scaling_stats_op=WEIGHT_SCALING_STATS_OP,
                      weight_scaling_per_output_channel=WEIGHT_SCALING_PER_OUTPUT_CHANNEL,
                      weight_restrict_scaling_type=WEIGHT_RESTRICT_SCALING_TYPE,
                      weight_narrow_range=WEIGHT_NARROW_RANGE,
                      weight_scaling_min_val=SCALING_MIN_VAL,
                      return_quant_tensor=WEIGHT_RETURN_QUANT_TENSOR):
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


def make_quant_avg_pool(bit_width,
                        kernel_size=POOL_KERNEL_SIZE,
                        stride=POOL_STRIDE,
                        signed=POOL_SIGNED):
    '''Helper for AveragePooling layers'''
    quant_type = get_quant_type(bit_width)
    return qnn.QuantAvgPool2d(kernel_size=kernel_size,
                              quant_type=quant_type,
                              signed=signed,
                              stride=stride,
                              min_overall_bit_width=1,
                              max_overall_bit_width=bit_width)
