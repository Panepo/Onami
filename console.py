from dotenv import load_dotenv
import os

load_dotenv()
backend = os.getenv("BACKEND")

if backend == "openvino":
  from modelsOV import pipe
elif backend == "openvino-genai":
  from modelsOVgen import pipe
else:
  raise ValueError(f"Unknown backend: {backend}")

import time

prompt = "You are a helpful assistant that identify the person, time, place and what happen in brief in the following article: "

def demo():
  while 1:
    print("================================================")
    text = input("Please say something: ")
    start_time = time.process_time()
    response = pipe.generate(text)
    print(response)
    end_time = time.process_time()
    elapsed_time = end_time - start_time
    print(f"Process time: {elapsed_time} seconds")

if __name__ == "__main__":
  demo()
