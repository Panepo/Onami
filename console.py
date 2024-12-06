import time

prompt = ""

def demo():
  start_time = time.time()

  from modelsOV import pipe, config, model_dir
  DEFAULT_SYSTEM_PROMPT = """\
  You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
  If a question does not make any sense or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
  """

  pipe.start_chat(system_message=DEFAULT_SYSTEM_PROMPT)

  import openvino_genai as ov_genai
  tokenizers = ov_genai.Tokenizer(model_dir)
  tokenizer_kwargs = {}

  load_time = time.time() - start_time
  print(f"Load time: {load_time} seconds")

  while 1:
    print("================================================")
    message = input("Please say something: ")
    time1 = time.time()
    input_tokens = tokenizers.encode(message, **tokenizer_kwargs)
    time2 = time.time()

    response = pipe.generate(message, config)
    print(response)
    time3 = time.time()

    encode_time = time2 - time1
    inference_time = time3 - time2
    print(f"Input length: {len(message)}, Output length: {len(response)}")
    print(f"Prompt eval time: {encode_time} seconds, speed: {len(message) / encode_time} chars/sec")
    print(f"Output eval time: {inference_time} seconds, speed: {len(response) / inference_time} chars/sec")

if __name__ == "__main__":
  demo()
