from dotenv import load_dotenv
import os

load_dotenv()
device = os.getenv("DEVICE")
model = os.getenv("MODEL")

if model == "llama3.2":
  model_dir = "models/Llama-3.2-3B-Instruct-ov-int4"
elif model == "tinyllama":
  model_dir = "models/TinyLlama-1.1B-Chat-v1.0-int4-ov"
elif model == "phi3":
  model_dir = "models/Phi-3-mini-4k-instruct-int4-ov"
else:
  raise ValueError(f"Unknown depth: {model}")

if device == "CPU":
  print("Current running on CPU")
elif device == "GPU":
  print("Current running on GPU")
elif device == "NPU":
  print("Current running on NPU")
else:
  raise ValueError(f"Unknown device: {device}")

import openvino_genai as ov_genai
from modelsConfig import model_configuration

tokenizers = ov_genai.Tokenizer(model_dir)
tokenizer_kwargs = {}
stop_token_ids = tokenizers.encode(model_configuration["stop_tokens"], **tokenizer_kwargs)

pipeline_config = { "stop_token_ids": stop_token_ids }

if device == "NPU":
  pipeline_config["NPUW_CACHE_DIR"] = ".npucache"
  pipe = ov_genai.LLMPipeline(str(model_dir), "NPU", pipeline_config)
else:
  pipe = ov_genai.LLMPipeline(str(model_dir), device)

config = pipe.get_generation_config()
config.temperature = 0.7
config.top_p = 0.95
config.top_k = 50
config.max_new_tokens = 2048
config.repetition_penalty = 1.1

if device == "NPU":
  config.do_sample = False
else:
  config.do_sample = config.temperature > 0.0


