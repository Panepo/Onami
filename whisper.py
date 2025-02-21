from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv()
device = os.getenv("DEVICE")
model = os.getenv("WMODEL")

model_dir = Path("models")
model_path = {
  "base": "whisper-base",
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

if model == "base":
  print(f"Current running on {model_path[model]} model")
else:
  raise ValueError(f"Unknown model: {model}")

import openvino_genai as ov_genai

target_dir = model_dir / model_path[model]
pipe = ov_genai.WhisperPipeline(target_dir, device)

config = pipe.get_generation_config()
config.max_new_tokens = 1024  # increase this based on your speech length
# 'task' and 'language' parameters are supported for multilingual models only
#config.language = "<|en|>"  # can switch to <|zh|> for Chinese language
#config.task = "transcribe"
config.return_timestamps = False
