from dotenv import load_dotenv
import os

load_dotenv()
device = os.getenv("DEVICE")
model = os.getenv("VMODEL")

from pathlib import Path
from PIL import Image
from io import BytesIO
from openvino import Tensor
import requests
import openvino_genai as ov_genai
import numpy as np
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

if model == "phi3.5vision":
  print(f"Current running on Phi-3.5 vision model")
  raise NotImplementedError("Phi-3.5 vision model is not supported yet")
elif model == "internvl2":
  print(f"Current running on {model_configuration['model_id']} model")
else:
  raise ValueError(f"Unknown vision model: {model}")

target_dir = model_dir / model_path[model]

def read_image(image: Image) -> Tensor:
  image_data = np.array(image).reshape(1, image.size[1], image.size[0], 3).astype(np.uint8)
  return Tensor(image_data)

def read_image_path(path: str) -> Tensor:
  if str(path).startswith("http") or str(path).startswith("https"):
    response = requests.get(path)
    image = Image.open(BytesIO(response.content)).convert("RGB")
  else:
    image = Image.open(path).convert("RGB")
  image_data = np.array(image.getdata()).reshape(1, image.size[1], image.size[0], 3).astype(np.byte)
  return image, Tensor(image_data)

enable_compile_cache = dict()
if "GPU" == device:
  # Cache compiled models on disk for GPU to save time on the
  # next run. It's not beneficial for CPU.
  enable_compile_cache["CACHE_DIR"] = "vlm_cache"

pipe = ov_genai.VLMPipeline(target_dir, device, **enable_compile_cache)

config = ov_genai.GenerationConfig()
config.max_new_tokens = 128

if "stop_strings" in model_configuration:
  config.stop_strings = set(model_configuration["stop_strings"])

if __name__ == "__main__":
  def streamer(subword: str) -> bool:
    print(subword, end="", flush=True)

  image_url = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11"
  image_file = Path("cat.png")

  if not image_file.exists():
    image, image_tensor = read_image_path(image_url)
    image.save(image_file)
  else:
    image, image_tensor = read_image_path(image_file)

  text_message = "What is unusual on this image?"

  prompt = text_message
  output = pipe.generate(prompt, image=image_tensor, generation_config=config, streamer=streamer)
  perf_metrics = output.perf_metrics

  print(f"Load time: {perf_metrics.get_load_time():.2f} ms")
  print(f"Generate time: {perf_metrics.get_generate_duration().mean:.2f} ± {perf_metrics.get_generate_duration().std:.2f} ms")
  print(f"Tokenization time: {perf_metrics.get_tokenization_duration().mean:.2f} ± {perf_metrics.get_tokenization_duration().std:.2f} ms")
  print(f"Detokenization time: {perf_metrics.get_detokenization_duration().mean:.2f} ± {perf_metrics.get_detokenization_duration().std:.2f} ms")
  print(f"Embeddings preparation time: {perf_metrics.get_prepare_embeddings_duration().mean:.2f} ± {perf_metrics.get_prepare_embeddings_duration().std:.2f} ms")
  print(f"TTFT: {perf_metrics.get_ttft().mean:.2f} ± {perf_metrics.get_ttft().std:.2f} ms")
  print(f"TPOT: {perf_metrics.get_tpot().mean:.2f} ± {perf_metrics.get_tpot().std:.2f} ms")
  print(f"Throughput : {perf_metrics.get_throughput().mean:.2f} ± {perf_metrics.get_throughput().std:.2f} tokens/s")

