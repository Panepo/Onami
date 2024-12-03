from modelsOV import pipe
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
