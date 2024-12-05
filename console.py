import time

prompt = "You are a helpful assistant that identify the person, time, place and what happen in brief in the following article: "

def demo():
  start_time = time.time()

  from modelsOV import pipe, config
  pipe.start_chat(system_message=prompt)

  load_time = time.time() - start_time
  print(f"Load time: {load_time} seconds")

  while 1:
    print("================================================")
    message = input("Please say something: ")
    start_time2 = time.time()
    response = pipe.generate(message, config)
    print(response)
    end_time = time.time()
    elapsed_time = end_time - start_time2
    print(f"Input length: {len(message)}, process time: {elapsed_time} seconds, speed: {len(message) / elapsed_time} chars/sec")

if __name__ == "__main__":
  demo()
