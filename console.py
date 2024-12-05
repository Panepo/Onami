import time

prompt = "You are a helpful assistant that identify the person, time, place and what happen in brief in the following article: "

def demo():
  start_time = time.time()

  from modelsOV import pipe, config, model_dir
  pipe.start_chat(system_message=prompt)

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
