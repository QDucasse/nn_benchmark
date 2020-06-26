# ===================================================================
# =========================== IMPORTS ===============================
# ===================================================================

import onnx
import brevitas.onnx as bo
from finn.util.test          import get_test_model_trained
from finn.core.modelwrapper  import ModelWrapper
from finn.custom_op.registry import getCustomOp

## Basic Transformations
from finn.transformation.general 	     import GiveReadableTensorNames, GiveUniqueNodeNames
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.infer_shapes 	 import InferShapes
from finn.transformation.fold_constants  import FoldConstants

## Streamline Transformations
import finn.transformation.streamline.absorb as absorb
from finn.transformation.streamline 				 import Streamline
from finn.transformation.bipolar_to_xnor 			 import ConvertBipolarMatMulToXnorPopcount
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds

## HLS Conversion
import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls
from finn.transformation.fpgadataflow.create_dataflow_partition import CreateDataflowPartition
from finn.transformation.fpgadataflow.insert_dwc 				import InsertDWC
from finn.transformation.fpgadataflow.insert_fifo 				import InsertFIFO
from finn.transformation.fpgadataflow.insert_tlastmarker 		import InsertTLastMarker

## Board Deployment
from finn.util.basic 										   import pynq_part_map
from finn.transformation.fpgadataflow.prepare_ip 		       import PrepareIP
from finn.transformation.fpgadataflow.hlssynth_ip 			   import HLSSynthIP
from finn.transformation.fpgadataflow.create_stitched_ip 	   import CreateStitchedIP
from finn.transformation.fpgadataflow.replace_verilog_relpaths import ReplaceVerilogRelPaths

from finn.transformation.fpgadataflow.make_pynq_proj   import MakePYNQProject
from finn.transformation.fpgadataflow.synth_pynq_proj  import SynthPYNQProject
from finn.transformation.fpgadataflow.make_pynq_driver import MakePYNQDriver
from finn.transformation.fpgadataflow.make_deployment  import DeployToPYNQ


# ===================================================================
# ===================================================================

build_dir = "/workspace/finn/onnx_experiments/tfc_example_onnx/""

def save(model,suffix):
	model.save(build_dir + "tfc_w1_a1_" + suffix + ".onnx")

def log(string):
	print("=================================")
	print("  "+string)
	print("=================================\n\n")


tfc = get_test_model_trained("TFC", 1, 1)
bo.export_finn_onnx(tfc, (1, 1, 28, 28), "./tfc_w1_a1.onnx")
model = ModelWrapper("./tfc_w1_a1.onnx")

# Trained
model = ModelWrapper("/workspace/finn/onnx_experiments/QuantTFC.onnx")

# Basic Transformations
model = model.transform(InferShapes())
model = model.transform(FoldConstants())
model = model.transform(GiveUniqueNodeNames())
model = model.transform(GiveReadableTensorNames())
log("Basic transformations completed")
save(model,"tidy")

# Streamline
model = model.transform(InferDataTypes())
model = model.transform(Streamline())
model = model.transform(ConvertBipolarMatMulToXnorPopcount())
model = model.transform(absorb.AbsorbAddIntoMultiThreshold())
model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
model = model.transform(RoundAndClipThresholds())
log("Streamline transformations completed")
save(model,"streamlined")

# HLS Conversion
model = model.transform(to_hls.InferBinaryStreamingFCLayer("decoupled"))
parent_model = model.transform(CreateDataflowPartition())
save(model,"dataflow")

sdp_node = getCustomOp(parent_model.graph.node[2])
dataflow_model_filename = sdp_node.get_nodeattr("model")
model = ModelWrapper(dataflow_model_filename)

fc_layers = model.get_nodes_by_op_type("StreamingFCLayer_Batch")
# (PE, SIMD, in_fifo_depth, out_fifo_depth, ramstyle) for each layer
config = [
    (16, 49, 16, 64, "block"),
    (8, 8, 64, 64, "auto"),
    (8, 8, 64, 64, "auto"),
    (10, 8, 64, 10, "distributed"),
]
for fcl, (pe, simd, ififo, ofifo, ramstyle) in zip(fc_layers, config):
    fcl_inst = getCustomOp(fcl)
    fcl_inst.set_nodeattr("PE", pe)
    fcl_inst.set_nodeattr("SIMD", simd)
    fcl_inst.set_nodeattr("inFIFODepth", ififo)
    fcl_inst.set_nodeattr("outFIFODepth", ofifo)
    fcl_inst.set_nodeattr("ram_style", ramstyle)

model = model.transform(InsertDWC())
model = model.transform(InsertFIFO())
model = model.transform(InsertTLastMarker())
log("HLS conversion transformations completed")
save(model,"fold_factors")


# Hardware Preparation
pynq_board = "Pynq-Z1"
fpga_part = pynq_part_map[pynq_board] # = xc7z020clg400-1
target_clk_ns = 10

# Generate IP
log("Preparing IP blocks generation")
model = model.transform(GiveUniqueNodeNames())
model = model.transform(PrepareIP(fpga_part, target_clk_ns))
log("IP blocks preparation completed")

log("Synthesizing IP blocks")
model = model.transform(HLSSynthIP())
model = model.transform(ReplaceVerilogRelPaths())
log("IP blocks synthesized")
save(model,"ip_blocks")


log("Stitching IP blocks")
model = model.transform(CreateStitchedIP(fpga_part))
log("IP blocks stitched")
save(model,"stitch")


# PYNQ Project, Driver and Deployment
log("PYNQ Project creation launched")
model = model.transform(MakePYNQProject(pynq_board))
log("PYNQ project created")

log("Synthesis, Place and Route launched")
model = model.transform(SynthPYNQProject())
log("Synthesis, Place and Route completed")
save(model,"post_synthesis")

log("Driver generation launched")
model = model.transform(MakePYNQDriver())
log("Driver generation completed")

log("Deployment launched")
ip = "192.168.2.99"
port = "22"
username = "xilinx"
password = "xilinx"
target_dir = "/home/xilinx/finn_tfc_end2end_example"
model = model.transform(DeployToPYNQ(ip, port, username, password, target_dir))
log("Deployment completed")
save(model,"deploy")

# Execution on the board and tests
from pkgutil import get_data
import onnx.numpy_helper as nph
import matplotlib.pyplot as plt

raw_i = get_data("finn", "data/onnx/mnist-conv/test_data_set_0/input_0.pb")
x = nph.to_array(onnx.load_tensor_from_string(raw_i))
# plt.imshow(x.reshape(28,28), cmap='gray') # Display an image of the MNIST dataset
build_dir = "/home/qducasse/Desktop/nn_projects/finn/onnx/tfc_w1_a1_"
parent_model = ModelWrapper(build_dir+"dataflow.onnx")
sdp_node = parent_model.graph.node[2]
remote_exec_model = build_dir + "deploy.onnx"
getCustomOp(sdp_node).set_nodeattr("model", remote_exec_model)
parent_model.save(model,"dataflow_parent_with_remote_bitfile_exec")

## EXECUTION

import numpy as np
from finn.core.onnx_exec import execute_onnx
iname = parent_model.graph.input[0].name
oname = parent_model.graph.output[0].name
ishape = parent_model.get_tensor_shape(iname)
input_dict = {iname: x.reshape(ishape)}
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
