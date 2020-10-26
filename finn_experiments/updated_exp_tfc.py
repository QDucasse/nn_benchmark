# ===================================================================
# =========================== IMPORTS ===============================
# ===================================================================

import argparse
import os
import sys

import onnx
import brevitas.onnx as bo
from finn.util.test          import get_test_model_trained
from finn.core.modelwrapper  import ModelWrapper
from finn.custom_op.registry import getCustomOp

## Basic Transformations
from finn.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames, RemoveStaticGraphInputs
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.fold_constants import FoldConstants

## Pre and Post Processing
from finn.util.pytorch import ToTensor
from finn.transformation.merge_onnx_models import MergeONNXModels
from finn.core.datatype import DataType
from finn.transformation.insert_topk import InsertTopK

## Streamline Transformations
import finn.transformation.streamline.absorb as absorb
from finn.transformation.streamline.reorder          import MoveScalarLinearPastInvariants
from finn.transformation.streamline                  import Streamline
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
from finn.transformation.infer_data_layouts          import InferDataLayouts
from finn.transformation.general                     import RemoveUnusedTensors

## HLS Conversion and synthesis
import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls
from finn.transformation.fpgadataflow.create_dataflow_partition import CreateDataflowPartition
from finn.transformation.fpgadataflow.make_zynq_proj            import ZynqBuild

## Board Deployment
from finn.util.basic import pynq_part_map
from finn.transformation.fpgadataflow.make_deployment import DeployToPYNQ


# ===================================================================
# HELPERS
# ===================================================================

def save(model,suffix):
    global name
    model.save(build_dir + name + "_" + suffix + ".onnx")

def load(suffix):
    global name
    return ModelWrapper(build_dir+ name + "_" + suffix + ".onnx")

def log(string):
    print("=================================")
    print("  "+string)
    print("=================================\n\n")

# ===================================================================
# TRANSFORMATIONS
# ===================================================================

# Basic Transformations
def tidy_up(model):
    log("Basic transformations launched")
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model = model.transform(RemoveStaticGraphInputs())
    log("Basic transformations completed")
    save(model,"0_tidy")
    return model

# Pre and post processing
def pre_processing(model):
    log("Starting Pre Processing")
    global_inp_name = model.graph.input[0].name
    ishape = model.get_tensor_shape(global_inp_name)
    # preprocessing: torchvision's ToTensor divides uint8 inputs by 255
    totensor_pyt = ToTensor()
    chkpt_preproc_name = build_dir+ "tfc_preproc.onnx"
    bo.export_finn_onnx(totensor_pyt, ishape, chkpt_preproc_name)
    # join preprocessing and core model
    pre_model = ModelWrapper(chkpt_preproc_name)
    model = model.transform(MergeONNXModels(pre_model))
    # add input quantization annotation: UINT8 for all BNN-PYNQ models
    global_inp_name = model.graph.input[0].name
    model.set_tensor_datatype(global_inp_name, DataType.UINT8)
    log("Finished Pre Processing!")
    save(model,"1_with_preproc")
    return model

def post_processing(model):
    log("Starting Post Processing")
    # Insert Top-1 node at the end
    model = model.transform(InsertTopK(k=1))
    # Tidy-up again
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model = model.transform(RemoveStaticGraphInputs())
    log("Finished Post Processing!")
    save(model,"2_with_pre_post")
    return model


# Streamline
def streamline(model, binary=True):
    log("Streamline transformations launched")
    model = model.transform(MoveScalarLinearPastInvariants())
    model = model.transform(Streamline())
    # Absorb add and mul in thresholds
    model = model.transform(absorb.AbsorbAddIntoMultiThreshold())
    model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
    # Absorb add-mul in top-k
    model = model.transform(absorb.AbsorbScalarMulAddIntoTopK())
    model = model.transform(RoundAndClipThresholds())
    # Tidy-up
    model = model.transform(InferDataLayouts())
    model = model.transform(RemoveUnusedTensors())
    log("Streamline transformations completed")
    save(model,"3_streamlined")
    return model

# HLS Conversion
# choose the memory mode for the MVTU units, decoupled or const
def hls_conversion(model, binary=True):
    log("HLS Conversion launched")
    mem_mode = "decoupled"
    if binary:
        model = model.transform(to_hls.InferBinaryStreamingFCLayer(mem_mode))
    model = model.transform(to_hls.InferQuantizedStreamingFCLayer(mem_mode))
    # TopK to LabelSelect
    model = model.transform(to_hls.InferLabelSelectLayer())
    # input quantization (if any) to standalone thresholding
    model = model.transform(to_hls.InferThresholdingLayer())
    log("HLS Conversion finished")
    save(model,"4_hls_conversion")
    return model

def create_dataflow_partition(model):
    log("Creating Dataflow Partition")
    parent_model = model.transform(CreateDataflowPartition())
    save(parent_model,"5_dataflow_parent")

    sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
    sdp_node = getCustomOp(sdp_node)
    dataflow_model_filename = sdp_node.get_nodeattr("model")
    dataflow_model = ModelWrapper(dataflow_model_filename)
    save(model,"5_dataflow_model")
    log("Dataflow partition created")
    return dataflow_model

def folding(model):
    log("Tuning folding")
    fc_layers = model.get_nodes_by_op_type("StreamingFCLayer_Batch")
    # (PE, SIMD, in_fifo_depth, out_fifo_depth, ramstyle) for each layer
    # Test Divided by two the PE and in_fifo_depth
    config = [
        # (2, 8, 2, 8, "block"),
        # (1, 1, 8, 8, "auto"),
        # (1, 1, 8, 8, "auto"),
        # (5, 1, 8, 5, "distributed"),
        (2, 2, 8, 8, "block"),
        (2, 2, 8, 8, "auto"),
        (2, 2, 8, 8, "auto"),
        (2, 2, 8, 8, "distributed"),
    ]
    for fcl, (pe, simd, ififo, ofifo, ramstyle) in zip(fc_layers, config):
        fcl_inst = getCustomOp(fcl)
        fcl_inst.set_nodeattr("PE", pe)
        fcl_inst.set_nodeattr("SIMD", simd)
        fcl_inst.set_nodeattr("inFIFODepth", ififo)
        fcl_inst.set_nodeattr("outFIFODepth", ofifo)
        fcl_inst.set_nodeattr("ram_style", ramstyle)
        print("MW="+str(fcl_inst.get_nodeattr("MW")))
        print("SIMD="+str(fcl_inst.get_nodeattr("SIMD")))
        print("---")
    log("Folding completed")
    save(model,"6_fold_factors")
    return model

# Hardware build
def create_IP_and_synthesis(model, platform, period_ns):
    log("Creating Project and synthesis")
    model = model.transform(ZynqBuild(platform = platform, period_ns = period_ns))
    log("Synthesis completed!")
    save(model,"7_post_synthesis")
    return model

def deploy(model, ip, port, username, password, target_dir):
    log("Deployment launched")
    model = model.transform(DeployToPYNQ(ip, port, username, password, target_dir))
    log("Deployment completed")
    save(model,"8_deploy")
    return model

if __name__ == "__main__":
    # Parse CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--acq", type=int)
    parser.add_argument("--weq", type=int)
    parser.add_argument("--inq", type=int)
    args = parser.parse_args()

    name = "QuantTFC_A{0}W{1}I{2}".format(args.acq, args.weq, args.inq)
    # Directory and model specification
    build_dir = "/workspace/finn/tested_networks/QuantTFC_A{0}W{1}I{2}/".format(args.acq, args.weq, args.inq)
    model = ModelWrapper("/workspace/finn/tested_networks/QuantTFC_A{0}W{1}I{2}/QuantTFC_A{0}W{1}I{2}_E40.onnx".format(args.acq, args.weq, args.inq))
    binary = (args.acq == 1)
    # Synthesis info
    pynq_board = "Pynq-Z1"
    fpga_part = pynq_part_map[pynq_board]
    target_clk_ns = 10
    # Deployment info
    ip = "192.168.2.99"
    port = "22"
    username = "xilinx"
    password = "xilinx"
    target_dir = "/home/xilinx/finn_tfc_experiment"

    # Transformations
    model = tidy_up(model)
    model = streamline(model, binary)
    model = hls_conversion(model, binary)
    model = create_dataflow_partition(model)
    model = folding(model)
    # Synthesis
    model = create_IP_and_synthesis(model, pynq_board, target_clk_ns)
    # model = load("7_post_synthesis")
    # PYNQ Deployment
    model = deploy(model, ip, port, username, password, target_dir)

    # ==========================================================================
    # TESTING EXECUTION ON BOARD
    # ==========================================================================

    # Execution on the board and tests
    from pkgutil import get_data
    import onnx.numpy_helper as nph
    import matplotlib.pyplot as plt
    from PIL import Image
    import torchvision.transforms.functional as TF
    import shutil
    import os

    # image = Image.open('/workspace/finn/onnx_experiments/img_MNIST_grayscale.png')
    # x = TF.to_tensor(image)
    # x.unsqueeze_(0)
    #
    # parent_model = load("5_dataflow_parent")
    # sdp_node = parent_model.graph.node[2]
    # remote_exec_model = build_dir + "8_deploy.onnx"
    # getCustomOp(sdp_node).set_nodeattr("model", remote_exec_model)
    # save(parent_model,"9_dataflow_parent_with_remote_bitfile_exec")

    # THROUGHPUT TESTS
    from finn.core.throughput_test import throughput_test_remote
    res = throughput_test_remote(model,batchsize=10000)
    print("Network metrics:")
    with open(build_dir+"res.txt","w") as f:
        for key in res:
            print(str(key) + ": " + str(res[key]))
            f.write(str(key) + ": " + str(res[key]) + "\n")


    II = 64
    # frequency in MHz
    f_MHz = 100
    # expected throughput in MFPS
    expected_throughput = f_MHz / II
    # measured throughput (FPS) from throughput test, converted to MFPS
    measured_throughput = res["throughput[images/s]"] * 0.000001
    # peformance
    print("We reach approximately " + str(round((measured_throughput / expected_throughput)*100)) + "% of the ideal performance.")

    # with open(build_dir+"res.txt","a") as f:
    #     f.write("We reach approximately " + str(round((measured_throughput / expected_throughput)*100)) + "% of the ideal performance." + "\n")
    #
    #
    # finn_dev_files = os.listdir("/tmp/finn_dev_quentin/")
    # for file in finn_dev_files:
    #     if file.startswith("vivado_zynq"):
    #         vivado_zynq_build = file
    # shutil.move("/tmp/finn_dev_quentin/"+vivado_zynq_build, build_dir+"vivado_proj/")
