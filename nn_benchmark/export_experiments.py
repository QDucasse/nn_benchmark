# -*- coding: utf-8 -*-

# nn_benchmark
# author - Quentin Ducasse
# https://github.com/QDucasse
# quentin.ducasse@ensta-bretagne.org

import sys
from nn_benchmark.core import ObjDict, Trainer
from nn_benchmark.networks import QuantTFC, QuantCNV, QuantMobilenetV1

if __name__ == "__main__":
    acq_list = [2, 4, 8, 16, 32]
    weq_list = [2, 4, 8, 16, 32]
    inq_list = [8, 8, 8, 32, 32]
    epochs   = [10, 20, 30, 40]

    exporter = Exporter()

    for epoch in epochs:
        for acq, weq, inq in zip(acq_list, weq_list, inq_list):
            # Load correct model
            cnv = QuantCNV()
            if epoch != 40:
                cnv_model = "/workspace/finn/trained_onnx/QuantCNV_A{0}W{1}I{2}/checkpoints/checkpoint_{3}.tar".format(acq, we1, inq, epoch)
            else:
                 cnv_model = "/workspace/finn/trained_onnx/QuantCNV_A{0}W{1}I{2}/checkpoints/best.tar".format(acq, we1, inq)
            # Generate ONNX counterpart
            output_path = "workspace/finn/onnx_experiments/QuantCNV_A{0}W{1}I{2}/QuantCNV_A{0}W{1}I{2}.onnx".
            exporter.export_onnx(model = cnv, output_dir_path = output_path, in_channels = 3,
                                 act_bit_width = acq, weight_bit_width = weq, input_bit_width = inq)
