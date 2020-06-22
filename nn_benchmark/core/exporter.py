# -*- coding: utf-8 -*-

# Included in:
# nn_benchmark
# author - Quentin Ducasse
# https://github.com/QDucasse
# quentin.ducasse@ensta-bretagne.org

# ONNX exporter

import torch
import brevitas.onnx as bo

class Exporter(object):

    def export_onnx(self, model, output_dir_path, act_bit_width=4, weight_bit_width=4, input_bit_width=8, in_channels=3, input_tensor=None, torch_onnx_kwargs={}):
        '''Export the model in ONNX format. A different export is provided is the
           network is a quantized one because the quantizations need to be stored as
           specific ONNX attributes'''
        if model.name.startswith("Quant"):
            self.quant_export(model=model, output_dir_path=output_dir_path,
                              act_bit_width=act_bit_width, weight_bit_width=weight_bit_width,
                              input_bit_width=input_bit_width, in_channels=in_channels,
                              input_tensor=input_tensor, torch_onnx_kwargs=torch_onnx_kwargs)
        else:
            self.base_export(model=model, output_dir_path=output_dir_path, in_channels=in_channels)


    def base_export(self, model, output_dir_path, in_channels=3):
        input = torch.ones([1, in_channels, 32, 32], dtype=torch.float32)
        torch.onnx.export(model, input, output_dir_path +"/"+ model.name + ".onnx")

    def quant_export(self, model, output_dir_path,
                     act_bit_width=4, weight_bit_width=4,
                     input_bit_width=8, in_channels=3,
                     input_tensor=None, torch_onnx_kwargs={}):
        model_act_bit_width    = "A" + str(act_bit_width)
        model_weight_bit_width = "W" + str(weight_bit_width)
        model_input_bit_width  = "I" + str(input_bit_width)
        model_name_with_attr   = "_".join([model.name,model_act_bit_width,model_weight_bit_width,model_input_bit_width])
        bo.export_finn_onnx(module=model,
                            input_shape=(1, in_channels, 32, 32),
                            export_path=output_dir_path +"/"+ model_name_with_attr + ".onnx",
                            input_t=input_tensor)

if __name__ == "__main__":
    # from finn.util.test import get_test_model_trained
    # export_path = 'tests/mobilenet.onnx'
    # mobilenet = get_test_model_trained("mobilenet", 4, 4)
    # input_tensor = torch.ones([1,3,224,224],dtype=torch.float32)
    # bo.export_finn_onnx(mobilenet, (1, 3, 224, 224), export_path, input_t=input_tensor)

    from nn_benchmark.networks import QuantTFC
    export_path = 'tests/QuantTFC.onnx'
    input_tensor = torch.ones([1,3,32,32],dtype=torch.float32)
    tfc = QuantTFC()
    bo.export_finn_onnx(tfc, (1, 3, 32, 32), export_path, input_t=input_tensor)
