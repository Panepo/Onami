from dotenv import load_dotenv
import os

load_dotenv()
device = os.getenv("DEVICE")
model = os.getenv("VMODEL")

import openvino_genai as ov_genai
from PIL import Image
from openvino import Tensor
import numpy as np
from vlm_config import model_dir, model_path

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

if model == "phi3.5vision":
  print(f"Current running on Phi-3.5 vision model")
else:
  raise ValueError(f"Unknown model: {model}")

target_dir = model_dir / model_path[model]

def read_image(image: Image) -> Tensor:
  image_data = np.array(image).reshape(1, image.size[1], image.size[0], 3).astype(np.uint8)
  return Tensor(image_data)

def read_image_path(path: str) -> Tensor:
  pic = Image.open(path).convert("RGB")
  image_data = np.array(pic.getdata()).reshape(1, pic.size[1], pic.size[0], 3).astype(np.uint8)
  return Tensor(image_data)

enable_compile_cache = dict()
if "GPU" == device:
  # Cache compiled models on disk for GPU to save time on the
  # next run. It's not beneficial for CPU.
  enable_compile_cache["CACHE_DIR"] = "vlm_cache"

pipe = ov_genai.VLMPipeline(target_dir, device, **enable_compile_cache)

config = ov_genai.GenerationConfig()
config.max_new_tokens = 512
