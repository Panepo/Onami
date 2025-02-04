from dotenv import load_dotenv
import os

load_dotenv()
device = os.getenv("DEVICE")
model = os.getenv("MODEL")

from llm_config import llama31_dir, llama32_dir, tinyllama_dir, phi3_dir, phi35_dir, deepseekr1_dir, deepseekr18_dir

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
  print("Current running on Llama-3.2 3B model")
  model_dir = llama32_dir
elif model == "llama3.1":
  print("Current running on Llama-3.1 8B model")
  model_dir = llama31_dir
elif model == "tinyllama":
  print("Current running on TinyLlama 1.5B model")
  model_dir = tinyllama_dir
elif model == "phi3":
  print("Current running on Phi-3 4B model")
  model_dir = phi3_dir
elif model == "phi3.5":
  print("Current running on Phi-3.5 4B model")
  model_dir = phi35_dir
elif model == "deepseekr1":
  print("Current running on DeepSeek-R1 1.5B model")
  model_dir = deepseekr1_dir
elif model == "deepseekr18":
  print("Current running on DeepSeek-R1 8B model")
  model_dir = deepseekr18_dir
else:
  raise ValueError(f"Unknown model: {model}")

import openvino_genai as ov_genai
from llm_config import model_configuration

#tokenizer = ov_genai.Tokenizer(model_dir)
#tokenizer_kwargs = {}
pipeline_config = {}

if device == "NPU":
  pipeline_config["NPUW_CACHE_DIR"] = ".npucache"
  pipeline_config["MAX_PROMPT_LEN"] = 1024
  pipeline_config["MIN_RESPONSE_LEN"] = 512
  pipe = ov_genai.LLMPipeline(str(model_dir), "NPU", pipeline_config)
else:
  pipe = ov_genai.LLMPipeline(str(model_dir), device)

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
