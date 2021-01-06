# -*- coding: utf-8 -*-

# nn_benchmark
# author - Quentin Ducasse
# https://github.com/QDucasse
# quentin.ducasse@ensta-bretagne.org

import sys
import torch
from nn_benchmark.core import Exporter
from nn_benchmark.networks import QuantTFC

if __name__ == "__main__":
    acq_list = [2, 3, 4, 5, 6, 7, 8, 16, 32]
    weq_list = [2, 3, 4, 5, 6, 7, 8, 16, 32]
    inq_list = [8, 8, 8, 8, 8, 8, 8, 32, 32]

    exporter = Exporter()

    # TFC
    for acq in acq_list:
        for weq in weq_list:
            if ((acq <= 8) and (weq <= 8)):
                inq = 8
            else:
                inq = 32
            # Load correct model
            tfc = QuantTFC(in_channels=1, weight_bit_width=weq, act_bit_width=acq, in_bit_width=inq)
            tfc_model = "/workspace/finn/trained_models/QuantTFC_A{0}W{1}I{2}/checkpoints/best.tar".format(acq, weq, inq)
            package = torch.load(tfc_model, map_location='cpu')
            model_state_dict = package['state_dict']
            tfc.load_state_dict(model_state_dict)
            # Generate ONNX counterpart
            output_path = "/workspace/finn/trained_onnx/QuantTFC_A{0}W{1}I{2}".format(acq, weq, inq)
            print("Exporting QuantTFC_A{0}W{1}I{2}.onnx".format(acq,weq,inq))
            exporter.export_onnx(model = tfc, output_dir_path = output_path, in_channels = 1,
                                 act_bit_width = acq, weight_bit_width = weq, input_bit_width = inq,
                                 epoch = 40)
