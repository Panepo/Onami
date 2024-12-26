from dotenv import load_dotenv
import os

load_dotenv()
device = os.getenv("DEVICE")
model = os.getenv("MODEL")

from model_download import llama32_dir, tinyllama_dir, phi3_dir, phi35_dir

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

from time import time as timer
t1 = timer()

from transformers import AutoTokenizer, TextIteratorStreamer
from intel_npu_acceleration_library import NPUModelForCausalLM
from intel_npu_acceleration_library.compiler import CompilerConfig
import torch
import threading
from model_config import model_configuration, DEFAULT_SYSTEM_PROMPT

compiler_conf = CompilerConfig(dtype=torch.float16)

model = NPUModelForCausalLM.from_pretrained(model_dir, compiler_conf, use_cache=True).eval()
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_default_system_prompt=True)
tokenizer.pad_token_id = tokenizer.eos_token_id

try:
  while 1:
    print("================================================")
    message = input("Please say something: ")
    if message == "exit" or message == "quit" or message == "":
      break

    start_message = model_configuration.get("start_message", DEFAULT_SYSTEM_PROMPT)
    console_message_start = model_configuration.get("console_message_start", "")
    console_message_end = model_configuration.get("console_message_end", "")
    inputs = start_message + console_message_start + message + console_message_end

    inputs = tokenizer(inputs, return_tensors="pt")
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = {
      "inputs": inputs.input_ids,
      "attention_mask": inputs.attention_mask,
      "max_new_tokens": 1024,
      "streamer": streamer
    }

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    generated_text = ""
    t2 = timer()
    for new_text in streamer:
      print(new_text, end="", flush=True)
      if generated_text == "":
        t3 = timer()
      generated_text += new_text

    t4 = timer()
    word_count = len(generated_text.split(" "))
    print(f"\n============================")
    print(f"Load model time = {t2 - t1:.1f}s")
    print(f"First token time = {t3 - t2:.1f}s")
    print(f"Inference time = {t4 - t3:.1f}s for {word_count} words.")
    print(f"Average {word_count/(t4 - t3):.1f} word/second.")

except KeyboardInterrupt:
  pass
