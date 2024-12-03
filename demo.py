from modelsOV import pipe
from demo_helper import make_demo

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
"""

DEFAULT_RAG_PROMPT = """\
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\
"""

model_configuration = {
  "model_id": "meta-llama/Llama-3.2-3B-Instruct",
  "start_message": DEFAULT_SYSTEM_PROMPT,
  "stop_tokens": ["<|eot_id|>"],
  "has_chat_template": True,
  "start_message": " <|start_header_id|>system<|end_header_id|>\n\n" + DEFAULT_SYSTEM_PROMPT + "<|eot_id|>",
  "history_template": "<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{assistant}<|eot_id|>",
  "current_message_template": "<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{assistant}",
  "rag_prompt_template": f"<|start_header_id|>system<|end_header_id|>\n\n{DEFAULT_RAG_PROMPT}<|eot_id|>"
  + """<|start_header_id|>user<|end_header_id|>


  Question: {input}
  Context: {context}
  Answer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>


  """,
  "completion_to_prompt": "",
}

demo = make_demo(pipe, model_configuration, "llama3.2", "English")

try:
    demo.launch(debug=True)
except Exception:
    demo.launch(debug=True, share=True)
