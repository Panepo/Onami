from dotenv import load_dotenv
import os

load_dotenv()
device = os.getenv("DEVICE")
model = os.getenv("MODEL")

if model == "llama3.2":
  model_dir = "models/Llama-3.2-3B-Instruct-ov-int4"
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

if __name__ == "__main__":
  import time

  test_string = """
  You are a helpful assistant that identify the person, time, place and what happen in brief in the following article.
  On first January 01, 2001 at 0310 hrs, I was dispatched to 962 loggerhead island drive in reference to a disturbance between the occupants. Upon arrival of OFC HEINZ and myself, we met with Vicki R.BRISTAL. She states she and her boyfriend were at a friend’s house. Earlier this evening and argued about child custody matters from her previous marriage. Buchan left the gathering went home. Upon BRISTOL’S arrival home she noticed some of her personal items and had been boxed up and placed by the front door. According to BRISTOL, she attempted to go into their bedroom and get some other belongings but was unable because BUCHAN had locked the door and would not let her enter. She also stated that incident did not involve any physical altercation.
  """
  start_time = time.process_time()
  print(pipe.generate(test_string))
  end_time = time.process_time()
  elapsed_time = end_time - start_time
  print(f"Process time: {elapsed_time} seconds")
