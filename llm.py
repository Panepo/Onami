from dotenv import load_dotenv
import os

load_dotenv()
device = os.getenv("DEVICE")
model = os.getenv("MODEL")

from model_download import llama32_dir, tinyllama_dir, phi3_dir, phi35_dir

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
  print("Current running on Llama-3.2 model")
  model_dir = llama32_dir
elif model == "tinyllama":
  print("Current running on TinyLlama model")
  model_dir = tinyllama_dir
elif model == "phi3":
  print("Current running on Phi-3 model")
  model_dir = phi3_dir
elif model == "phi3.5":
  print("Current running on Phi-3.5 model")
  model_dir = phi35_dir
else:
  raise ValueError(f"Unknown model: {model}")

import openvino_genai as ov_genai
from model_config import model_configuration

tokenizer = ov_genai.Tokenizer(model_dir)
tokenizer_kwargs = {}
pipeline_config = {}

#stop_token = model_configuration.get("stop_tokens", "")
#if (len(stop_token) > 0):
#  stop_token_ids = tokenizer.encode(stop_token, **tokenizer_kwargs)
#  pipeline_config["stop_token_ids"] = stop_token_ids

if device == "NPU":
  pipeline_config["NPUW_CACHE_DIR"] = ".npucache"
  pipeline_config["MAX_PROMPT_LEN"] = 1024
  pipeline_config["MIN_RESPONSE_LEN"] = 512
  pipe = ov_genai.LLMPipeline(str(model_dir), "NPU", pipeline_config)
else:
  pipe = ov_genai.LLMPipeline(str(model_dir), device)

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


