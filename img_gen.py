from dotenv import load_dotenv
import os

load_dotenv()
device = os.getenv("DEVICE")

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

import openvino_genai as ov_genai
from llm_config import model_dir, model_path

target_dir = model_dir / model_path['stable-diffusion']
pipe = ov_genai.Image2ImagePipeline(target_dir, device)
