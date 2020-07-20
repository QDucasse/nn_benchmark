# -*- coding: utf-8 -*-

# nn_benchmark
# author - Quentin Ducasse
# https://github.com/QDucasse
# quentin.ducasse@ensta-bretagne.org

import sys
import torch
from nn_benchmark.core import Exporter
from nn_benchmark.networks import QuantTFC, QuantCNV

if __name__ == "__main__":
    acq_list = [2, 4, 8, 16, 32]
    weq_list = [2, 4, 8, 16, 32]
    inq_list = [8, 8, 8, 32, 32]
    epochs   = [10, 20, 30, 40]

    exporter = Exporter()

    tfc = QuantTFC(in_channels=1)
    tfc_model = "/workspace/finn/trained_onnx/QuantTFC_A32W32I32_20200720/checkpoints/best.tar"
    package = torch.load(tfc_model, map_location='cpu')
    model_state_dict = package['state_dict']
    tfc.load_state_dict(model_state_dict)
    # Generate ONNX counterpart
    output_path = "/workspace/finn/onnx_experiments/QuantTFC_A32W32I32_v2"
    print("Exporting QuantTFC_A32W32I32.onnx")
    exporter.export_onnx(model = tfc, output_dir_path = output_path, in_channels = 1,
                         act_bit_width = 32, weight_bit_width = 32, input_bit_width = 32,
                         epoch = 100)

    # TFC
    # for acq, weq, inq in zip(acq_list, weq_list, inq_list):
    #     # Load correct model
    #     tfc = QuantTFC(in_channels=1)
    #     tfc_model = "/workspace/finn/trained_onnx/QuantTFC_A{0}W{1}I{2}/checkpoints/best.tar".format(acq, weq, inq)
    #     package = torch.load(tfc_model, map_location='cpu')
    #     model_state_dict = package['state_dict']
    #     tfc.load_state_dict(model_state_dict)
    #     # Generate ONNX counterpart
    #     output_path = "/workspace/finn/onnx_experiments/QuantTFC_A{0}W{1}I{2}".format(acq, weq, inq)
    #     print("Exporting QuantTFC_A{0}W{1}I{2}_E{3}.onnx".format(acq,weq,inq,epoch))
    #     exporter.export_onnx(model = tfc, output_dir_path = output_path, in_channels = 1,
    #                          act_bit_width = acq, weight_bit_width = weq, input_bit_width = inq,
    #                          epoch = 100)


    # CNV
    # for epoch in epochs:
    #     for acq, weq, inq in zip(acq_list, weq_list, inq_list):
    #         # Load correct model
    #         cnv = QuantCNV()
    #         if epoch != 40:
    #             cnv_model = "/workspace/finn/trained_onnx/QuantCNV_A{0}W{1}I{2}/checkpoints/checkpoint_{3}.tar".format(acq, weq, inq, epoch)
    #         else:
    #              cnv_model = "/workspace/finn/trained_onnx/QuantCNV_A{0}W{1}I{2}/checkpoints/best.tar".format(acq, weq, inq)
    #         package = torch.load(cnv_model, map_location='cpu')
    #         model_state_dict = package['state_dict']
    #         cnv.load_state_dict(model_state_dict)
    #         # Generate ONNX counterpart
    #         output_path = "/workspace/finn/onnx_experiments/QuantCNV_A{0}W{1}I{2}".format(acq, weq, inq, epoch)
    #         print("Exporting QuantCNV_A{0}W{1}I{2}_E{3}.onnx".format(acq,weq,inq,epoch))
    #         exporter.export_onnx(model = cnv, output_dir_path = output_path, in_channels = 3,
    #                              act_bit_width = acq, weight_bit_width = weq, input_bit_width = inq,
    #                              epoch = epoch)
