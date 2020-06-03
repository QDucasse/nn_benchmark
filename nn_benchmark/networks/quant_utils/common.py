# -*- coding: utf-8 -*-

# nn_benchmark
# author - Quentin Ducasse
# https://github.com/QDucasse
# quentin.ducasse@ensta-bretagne.org

# Quantized layers helpers

from brevitas.core.quant import QuantType

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
