from dotenv import load_dotenv
import os

load_dotenv()
device = os.getenv("DEVICE")
model = os.getenv("MODEL")
draft = os.getenv("DRAFT")

import openvino_genai as ov_genai
from llm_config import model_dir, model_path, model_configuration

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

if model == "llama3.2":
  print(f"Current running on {model_configuration['model_id']} model")
elif model == "llama3.1":
  print(f"Current running on {model_configuration['model_id']} model")
else:
  raise ValueError(f"Unknown model: {model}")

target_dir = model_dir / model_path[model]

if draft == "llama3.1draft":
  print(f"Current running draft model on llama3.1draft model")
else:
  raise ValueError(f"Unknown model: {model}")

draft_dir = model_dir / model_path[draft]

draft_model = ov_genai.draft_model(str(draft_dir), device)

scheduler_config = ov_genai.SchedulerConfig()
scheduler_config.cache_size = 2

pipe = ov_genai.LLMPipeline(str(target_dir), device, scheduler_config=scheduler_config, draft_model=draft_model)

config = ov_genai.GenerationConfig()
config.max_new_tokens = 2048
config.temperature = 0.1
config.top_p = 1.0
config.top_k = 50
config.repetition_penalty = 1.1

if device == "NPU":
  config.do_sample = False
else:
  config.do_sample = config.temperature > 0.0

if "stop_strings" in model_configuration:
  config.stop_strings = set(model_configuration["stop_strings"])
