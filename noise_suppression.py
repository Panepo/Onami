from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv()
device = os.getenv("DEVICE")
model = os.getenv("NMODEL")

model_dir = Path("models")
model_path = {
  "denseunet": "noise-suppression-denseunet-ll-0001\\FP32\\noise-suppression-denseunet-ll-0001.xml",
  "poconetlike": "noise-suppression-poconetlike-0001\\FP32\\noise-suppression-poconetlike-0001.xml"
}

if device == "CPU":
  print("Current running on CPU")
elif device == "GPU":
  print("Current running on GPU")
elif device == "GPU.1":
  print("Current running on GPU.1")
elif device == "NPU":
  print("Current running on NPU")
else:
  raise ValueError(f"Unknown device: {device}")

if model == "denseunet":
  print(f"Current running on {model_path[model]} model")
elif model == "poconetlike":
  print(f"Current running on {model_path[model]} model")
else:
  raise ValueError(f"Unknown model: {model}")

from openvino import Core
import numpy as np

core = Core()

target_dir = model_dir / model_path[model]
ov_encoder = core.read_model(target_dir)

inp_shapes = {name: obj.shape for obj in ov_encoder.inputs for name in obj.get_names()}
out_shapes = {name: obj.shape for obj in ov_encoder.outputs for name in obj.get_names()}

state_out_names = [n for n in out_shapes.keys() if "state" in n]
state_inp_names = [n for n in inp_shapes.keys() if "state" in n]
if len(state_inp_names) != len(state_out_names):
    raise RuntimeError(
        "Number of input states of the model ({}) is not equal to number of output states({})".
            format(len(state_inp_names), len(state_out_names)))

state_param_num = sum(np.prod(inp_shapes[n]) for n in state_inp_names)
print("State_param_num = {} ({:.1f}Mb)".format(state_param_num, state_param_num*4e-6))

# load model to the device
compiled_model = core.compile_model(ov_encoder, device)
