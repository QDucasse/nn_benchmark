# -*- coding: utf-8 -*-

# nn_benchmark
# author - Quentin Ducasse
# https://github.com/QDucasse
# quentin.ducasse@ensta-bretagne.org

# Quantized activation layers helpers

import brevitas.nn as qnn
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.scaling      import ScalingImplType

from nn_benchmark.networks.quant_utils import get_quant_type

# Global quantization
SCALING_MIN_VAL   = 2e-16
ENABLE_BIAS_QUANT = False
MIN_OVERALL_BW = 2
MAX_OVERALL_BW = None

# Activation function quantization
ACT_SCALING_IMPL_TYPE               = ScalingImplType.PARAMETER
ACT_SCALING_PER_CHANNEL             = False
ACT_SCALING_RESTRICT_SCALING_TYPE   = RestrictValueType.LOG_FP
ACT_MAX_VAL                         = 6.0
ACT_RETURN_QUANT_TENSOR             = False
ACT_PER_CHANNEL_BROADCASTABLE_SHAPE = None

# Hard tanh
HARD_TANH_THRESHOLD = 1.0

def make_quant_relu(bit_width,
                    scaling_impl_type               = ACT_SCALING_IMPL_TYPE,
                    scaling_per_channel             = ACT_SCALING_PER_CHANNEL,
                    restrict_scaling_type           = ACT_SCALING_RESTRICT_SCALING_TYPE,
                    scaling_min_val                 = SCALING_MIN_VAL,
                    max_val                         = ACT_MAX_VAL,
                    min_overall_bit_width           = MIN_OVERALL_BW,
                    max_overall_bit_width           = MAX_OVERALL_BW,
                    return_quant_tensor             = ACT_RETURN_QUANT_TENSOR,
                    per_channel_broadcastable_shape = ACT_PER_CHANNEL_BROADCASTABLE_SHAPE):
    '''Helper for ReLU activation layers'''
    quant_type = get_quant_type(bit_width)
    return qnn.QuantReLU(bit_width                       = bit_width,
                         quant_type                      = quant_type,
                         scaling_impl_type               = scaling_impl_type,
                         scaling_per_channel             = scaling_per_channel,
                         restrict_scaling_type           = restrict_scaling_type,
                         scaling_min_val                 = scaling_min_val,
                         max_val                         = max_val,
                         min_overall_bit_width           = min_overall_bit_width,
                         max_overall_bit_width           = max_overall_bit_width,
                         return_quant_tensor             = return_quant_tensor,
                         per_channel_broadcastable_shape = per_channel_broadcastable_shape)

def make_quant_tanh(bit_width,
                    restrict_scaling_type = ACT_SCALING_RESTRICT_SCALING_TYPE,
                    min_overall_bit_width = MIN_OVERALL_BW,
                    max_overall_bit_width = MAX_OVERALL_BW,
                    return_quant_tensor   = ACT_RETURN_QUANT_TENSOR):
    '''Helper for tanh activation layers'''
    quant_type = get_quant_type(bit_width)
    return qnn.QuantTanh(bit_width             = bit_width,
                         quant_type            = quant_type,
                         restrict_scaling_type = restrict_scaling_type,
                         min_overall_bit_width = min_overall_bit_width,
                         max_overall_bit_width = max_overall_bit_width,
                         return_quant_tensor   = return_quant_tensor)

def make_quant_sigmoid(bit_width,
                       restrict_scaling_type = ACT_SCALING_RESTRICT_SCALING_TYPE,
                       min_overall_bit_width = MIN_OVERALL_BW,
                       max_overall_bit_width = MAX_OVERALL_BW,
                       return_quant_tensor   = ACT_RETURN_QUANT_TENSOR):
    '''Helper for sigmoid activation layers'''
    quant_type = get_quant_type(bit_width)
    return qnn.QuantSigmoid(bit_width             = bit_width,
                            quant_type            = quant_type,
                            min_overall_bit_width = min_overall_bit_width,
                            max_overall_bit_width = max_overall_bit_width,
                            restrict_scaling_type = restrict_scaling_type,
                            return_quant_tensor   = return_quant_tensor)

def make_quant_hard_tanh(bit_width,
                         scaling_impl_type               = ACT_SCALING_IMPL_TYPE,
                         scaling_per_channel             = ACT_SCALING_PER_CHANNEL,
                         restrict_scaling_type           = ACT_SCALING_RESTRICT_SCALING_TYPE,
                         scaling_min_val                 = SCALING_MIN_VAL,
                         threshold                       = HARD_TANH_THRESHOLD,
                         min_overall_bit_width           = MIN_OVERALL_BW,
                         max_overall_bit_width           = MAX_OVERALL_BW,
                         return_quant_tensor             = ACT_RETURN_QUANT_TENSOR,
                         per_channel_broadcastable_shape = ACT_PER_CHANNEL_BROADCASTABLE_SHAPE):
    '''Helper for Hard Tanh activation layers'''
    quant_type = get_quant_type(bit_width)
    return qnn.QuantHardTanh(bit_width=bit_width,
                             quant_type=quant_type,
                             scaling_per_channel             = scaling_per_channel,
                             scaling_impl_type               = scaling_impl_type,
                             restrict_scaling_type           = restrict_scaling_type,
                             scaling_min_val                 = scaling_min_val,
                             max_val                         = threshold,
                             min_val                         = -threshold,
                             min_overall_bit_width           = min_overall_bit_width,
                             max_overall_bit_width           = max_overall_bit_width,
                             per_channel_broadcastable_shape = per_channel_broadcastable_shape,
                             return_quant_tensor             = return_quant_tensor)
