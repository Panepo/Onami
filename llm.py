from dotenv import load_dotenv
import os

load_dotenv()
device = os.getenv("DEVICE")
model = os.getenv("MODEL")

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
elif model == "tinyllama":
  print(f"Current running on {model_configuration['model_id']} model")
elif model == "phi3":
  print(f"Current running on {model_configuration['model_id']} model")
elif model == "phi3.5":
  print(f"Current running on {model_configuration['model_id']} model")
elif model == "phi4":
  print(f"Current running on {model_configuration['model_id']} model")
elif model == "deepseekr1":
  print(f"Current running on {model_configuration['model_id']} model")
elif model == "deepseekr18":
  print(f"Current running on {model_configuration['model_id']} model")
elif model == "gemma2":
  print(f"Current running on {model_configuration['model_id']} model")
elif model == "gemma29":
  print(f"Current running on {model_configuration['model_id']} model")
else:
  raise ValueError(f"Unknown model: {model}")

target_dir = model_dir / model_path[model]

import openvino_genai as ov_genai
from llm_config import model_configuration

pipeline_config = {}

if device == "NPU":
  pipeline_config["NPUW_CACHE_DIR"] = ".npucache"
  pipeline_config["MAX_PROMPT_LEN"] = 1024
  pipeline_config["MIN_RESPONSE_LEN"] = 512
  pipe = ov_genai.LLMPipeline(str(target_dir), "NPU", pipeline_config)
else:
  pipe = ov_genai.LLMPipeline(str(target_dir), device)

if "genai_chat_template" in model_configuration:
  pipe.get_tokenizer().set_chat_template(model_configuration["genai_chat_template"])

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
