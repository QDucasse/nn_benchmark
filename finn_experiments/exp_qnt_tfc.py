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
from finn.transformation.double_to_single_float import DoubleToSingleFloat
from finn.transformation.general                 import GiveReadableTensorNames, GiveUniqueNodeNames # RemoveStaticGraphInputs
from finn.transformation.infer_datatypes        import InferDataTypes
from finn.transformation.infer_shapes             import InferShapes
from finn.transformation.fold_constants         import FoldConstants

## Streamline Transformations
import finn.transformation.streamline.absorb as absorb
from finn.transformation.lower_convs_to_matmul       import LowerConvsToMatMul
from finn.transformation.streamline                  import Streamline
from finn.transformation.streamline.reorder          import MakeMaxPoolNHWC
from finn.transformation.bipolar_to_xnor              import ConvertBipolarMatMulToXnorPopcount
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds

## HLS Conversion
import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls
from finn.transformation.move_reshape                           import RemoveCNVtoFCFlatten
from finn.transformation.fpgadataflow.create_dataflow_partition import CreateDataflowPartition
from finn.transformation.fpgadataflow.insert_dwc                 import InsertDWC
from finn.transformation.fpgadataflow.insert_fifo                 import InsertFIFO
from finn.transformation.fpgadataflow.insert_tlastmarker         import InsertTLastMarker

## Board Deployment
from finn.util.basic                                           import pynq_part_map
from finn.transformation.fpgadataflow.prepare_ip               import PrepareIP
from finn.transformation.fpgadataflow.hlssynth_ip              import HLSSynthIP
from finn.transformation.fpgadataflow.create_stitched_ip       import CreateStitchedIP
from finn.transformation.fpgadataflow.replace_verilog_relpaths import ReplaceVerilogRelPaths

from finn.transformation.fpgadataflow.make_pynq_proj   import MakePYNQProject
from finn.transformation.fpgadataflow.synth_pynq_proj  import SynthPYNQProject
from finn.transformation.fpgadataflow.make_pynq_driver import MakePYNQDriver
from finn.transformation.fpgadataflow.make_deployment  import DeployToPYNQ


# ===================================================================
# HELPERS
# ===================================================================

def save(model,suffix):
    model.save(build_dir + "tfc_" + suffix + ".onnx")

def load(suffix):
    return ModelWrapper(build_dir+"tfc_" + suffix + ".onnx")

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
    model = model.transform(DoubleToSingleFloat())
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    # model = model.transform(RemoveStaticGraphInputs())
    log("Basic transformations completed")
    save(model,"tidy")
    return model

# Streamline
def streamline(model, binary=True):
    log("Streamline transformations launched")
    model = model.transform(Streamline())
    # model = model.transform(RemoveUnusedTensors())
    log("Streamline transformations completed")
    save(model,"streamlined")
    return model

# HLS Conversion
# choose the memory mode for the MVTU units, decoupled or const
def hls_conversion(model, binary=True):
    log("HLS Conversion launched")
    mem_mode = "decoupled"
    if binary:
        model = model.transform(to_hls.InferBinaryStreamingFCLayer(mem_mode))
    model = model.transform(to_hls.InferQuantizedStreamingFCLayer(mem_mode))
    log("HLS Conversion finished")
    save(model,"hls_conversion")
    return model

def create_dataflow_partition(model):
    log("Creating Dataflow Partition")
    parent_model = model.transform(CreateDataflowPartition())
    save(parent_model,"dataflow_parent")

    sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
    sdp_node = getCustomOp(sdp_node)
    dataflow_model_filename = sdp_node.get_nodeattr("model")
    dataflow_model = ModelWrapper(dataflow_model_filename)
    save(model,"dataflow_model")
    log("Dataflow partition created")
    return dataflow_model

def folding(model):
    log("Tuning folding")
    fc_layers = model.get_nodes_by_op_type("StreamingFCLayer_Batch")
    # (PE, SIMD, in_fifo_depth, out_fifo_depth, ramstyle) for each layer
    # Test Divided by two the PE and in_fifo_depth
    config = [
        (8, 32, 8, 32, "block"),
        (4, 4, 32, 32, "auto"),
        (4, 4, 32, 32, "auto"),
        (5, 4, 32, 10, "distributed"),
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
    model = model.transform(InsertDWC())
    model = model.transform(InsertFIFO())
    model = model.transform(InsertTLastMarker())
    model = model.transform(GiveUniqueNodeNames())
    log("Folding completed")
    save(model,"fold_factors")
    return model

# Generate IP
def prepare_ip(model, fpga_part, target_clk_ns):
    log("Preparing IP blocks generation")
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(fpga_part, target_clk_ns))
    save(model,"ip_preparation")
    log("IP blocks preparation completed")
    return model

def synthetize_ip(model):
    log("Synthesizing IP blocks")
    model = model.transform(HLSSynthIP())
    model = model.transform(ReplaceVerilogRelPaths())
    log("IP blocks synthesized")
    save(model,"ip_blocks")
    return model

def stitch_ip(model, fpga_part):
    log("Stitching IP blocks")
    model = model.transform(CreateStitchedIP(fpga_part))
    log("IP blocks stitched")
    save(model,"stitch")
    return model

# PYNQ Project, Driver and Deployment
def create_project(model, pynq_board):
    log("PYNQ Project creation launched")
    model = model.transform(MakePYNQProject(pynq_board))
    log("PYNQ project created")
    vivado_proj = model.get_metadata_prop("vivado_pynq_proj")
    print("Vivado synthesis project is at %s/resizer.xpr" % vivado_proj)
    return model

def synthesis(model):
    log("Synthesis, Place and Route launched")
    model = model.transform(SynthPYNQProject())
    log("Synthesis, Place and Route completed")
    save(model,"post_synthesis")
    return model

def gen_driver(model):
    log("Driver generation launched")
    model = model.transform(MakePYNQDriver())
    log("Driver generation completed")
    return model

def deploy(model, ip, port, username, password, target_dir):
    log("Deployment launched")

    model = model.transform(DeployToPYNQ(ip, port, username, password, target_dir))
    log("Deployment completed")
    save(model,"deploy")
    return model

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


if __name__ == "__main__":
    # Parse CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--acq", type=int)
    parser.add_argument("--weq", type=int)
    parser.add_argument("--inq", type=int)
    parser.add_argument("--epoch", type=int)
    args = parser.parse_args()

    # Directory and model specification
    build_dir = "/workspace/finn/onnx_experiments/QuantTFC_A{0}W{1}I{2}/".format(args.acq, args.weq, args.inq)
    model = ModelWrapper("/workspace/finn/onnx_experiments/QuantTFC_A{0}W{1}I{2}/QuantTFC_A{0}W{1}I{2}_E{3}.onnx".format(args.acq, args.weq, args.inq, args.epoch))
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
    model = prepare_ip(model, fpga_part, target_clk_ns)
    model = synthetize_ip(model)
    model = stitch_ip(model, fpga_part)
    # PYNQ Deployment
    model = create_project(model, pynq_board)
    model = synthesis(model)
    model = gen_driver(model)
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

    image = Image.open('/workspace/finn/onnx_experiments/img_MNIST_grayscale.png')
    x = TF.to_tensor(image)
    x.unsqueeze_(0)

    parent_model = load("dataflow_parent")
    sdp_node = parent_model.graph.node[2]
    remote_exec_model = build_dir + "tfc_deploy.onnx"
    getCustomOp(sdp_node).set_nodeattr("model", remote_exec_model)
    save(parent_model,"dataflow_parent_with_remote_bitfile_exec")


    ## EXECUTION

    import numpy as np
    from finn.core.onnx_exec import execute_onnx
    iname = parent_model.graph.input[0].name
    oname = parent_model.graph.output[0].name
    ishape = parent_model.get_tensor_shape(iname)
    input_dict = {iname: x.numpy()[0].reshape(ishape)}
    ret = execute_onnx(parent_model, input_dict, True)

    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    logits = ret[oname].flatten()
    prob = softmax(logits)

    plt.bar(np.arange(10), prob)


    # THROUGHPUT TESTS
    from finn.core.throughput_test import throughput_test

    child_model = ModelWrapper(getCustomOp(sdp_node).get_nodeattr("model"))
    res = throughput_test(child_model)
    print("Network metrics:")
    for key in res:
        print(str(key) + ": " + str(res[key]))

    II = 64
    # frequency in MHz
    f_MHz = 100
    # expected throughput in MFPS
    expected_throughput = f_MHz / II
    # measured throughput (FPS) from throughput test, converted to MFPS
    measured_throughput = res["throughput[images/s]"] * 0.000001
    # peformance
    print("We reach approximately " + str(round((measured_throughput / expected_throughput)*100)) + "% of the ideal performance.")
