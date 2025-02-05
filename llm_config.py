from dotenv import load_dotenv
import os

load_dotenv()
model = os.getenv("MODEL")

from pathlib import Path

model_dir = Path("models")

llama31_dir = model_dir / "Llama-3.1-8B-Instruct-openvino-4bit"
llama32_dir = model_dir / "Llama-3.2-3B-Instruct-openvino-4bit"
tinyllama_dir = model_dir / "TinyLlama-1.1B-Chat-v1.0-int4-ov"
phi35_dir = model_dir / "Phi-3.5-4k-instruct-int4-ov"
phi3_dir = model_dir / "Phi-3-mini-4k-instruct-int4-ov"
deepseekr1_dir = model_dir / "DeepSeek-R1-Distill-Qwen-1.5B"
deepseekr18_dir = model_dir / "DeepSeek-R1-Distill-Llama-8B"

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
"""

DEFAULT_RAG_PROMPT = """\
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\
"""

def phi_completion_to_prompt(completion):
    return f"<|system|><|end|><|user|>{completion}<|end|><|assistant|>\n"


def llama3_completion_to_prompt(completion):
    return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{completion}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

if model == "llama3.2":
  model_configuration = {
    "model_id": "meta-llama/Llama-3.2-3B-Instruct",
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
    "completion_to_prompt": llama3_completion_to_prompt,
    "stop_strings": ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]
  }
elif model == "llama3.1":
   model_configuration = {
    "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
    "completion_to_prompt": llama3_completion_to_prompt,
    "stop_strings": ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]
  }
elif model == "phi3":
  model_configuration = {
    "model_id": "microsoft/Phi-3-mini-4k-instruct",
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
    "completion_to_prompt": phi_completion_to_prompt,
    "stop_strings": ["<|system|>", "<|user|>", "<|end|>", "<|assistant|>"]
  }
elif model == "tinyllama":
  model_configuration = {
    "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
  }
elif model == "phi3.5":
  model_configuration = {
    "model_id": "microsoft/Phi-3.5-mini-instruct",
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
    "completion_to_prompt": phi_completion_to_prompt,
    "stop_strings": ["<|system|>", "<|user|>", "<|end|>", "<|assistant|>"]
  }
elif model == "deepseekr1":
   model_configuration = {
    "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "genai_chat_template": "{% for message in messages %}{% if loop.first %}{{ '<｜begin▁of▁sentence｜>' }}{% endif %}{% if message['role'] == 'system' and message['content'] %}{{ message['content'] }}{% elif message['role'] == 'user' %}{{  '<｜User｜>' +  message['content'] }}{% elif message['role'] == 'assistant' %}{{ '<｜Assistant｜>' +  message['content'] + '<｜end▁of▁sentence｜>' }}{% endif %}{% if loop.last and add_generation_prompt and message['role'] != 'assitant' %}{{ '<｜Assistant｜>' }}{% endif %}{% endfor %}",
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
    "stop_strings": ["<｜end▁of▁sentence｜>", "<｜User｜>", "</User|>", "<|User|>", "<|end_of_sentence|>", "</｜", "｜end▁of▁sentence｜>"],
  }
elif model == "deepseekr18":
   model_configuration = {
    "model_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "genai_chat_template": "{% for message in messages %}{% if loop.first %}{{ '<｜begin▁of▁sentence｜>' }}{% endif %}{% if message['role'] == 'system' and message['content'] %}{{ message['content'] }}{% elif message['role'] == 'user' %}{{  '<｜User｜>' +  message['content'] }}{% elif message['role'] == 'assistant' %}{{ '<｜Assistant｜>' +  message['content'] + '<｜end▁of▁sentence｜>' }}{% endif %}{% if loop.last and add_generation_prompt and message['role'] != 'assitant' %}{{ '<｜Assistant｜>' }}{% endif %}{% endfor %}",
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
    "stop_strings": ["<｜end▁of▁sentence｜>", "<｜User｜>", "</User|>", "<|User|>", "<|end_of_sentence|>", "</｜", "｜end▁of▁sentence｜>"],
  }
else:
  raise ValueError(f"Unknown model: {model}")
