import gradio as gr
from threading import Event, Thread
from PIL import Image
from queue import Queue
from vlm import pipe, config

class TextQueue:
  def __init__(self) -> None:
    self.text_queue = Queue()
    self.stop_signal = None
    self.stop_tokens = []

  def __call__(self, text):
    self.text_queue.put(text)

  def __iter__(self):
    return self

  def __next__(self):
    value = self.text_queue.get()
    if value == self.stop_signal or value in self.stop_tokens:
      raise StopIteration()
    else:
      return value

  def reset(self):
    self.text_queue = Queue()

  def end(self):
    self.text_queue.put(self.stop_signal)

def fn_camera(image):
  return image.transpose(Image.FLIP_LEFT_RIGHT)

def fn_llm(image, question):
  if image is None:
    return "Please upload an image"
  elif question is None:
    return "Please enter a question"
  else:
    streamer = TextQueue()
    stream_complete = Event()

    def generate_and_signal_complete():
      streamer.reset()
      generation_kwargs = {"prompt": question, "generation_config": config, "streamer": streamer}
      if image is not None:
          generation_kwargs["image"] = image
      pipe.generate(**generation_kwargs)
      stream_complete.set()
      streamer.end()


    t1 = Thread(target=generate_and_signal_complete)
    t1.start()

    buffer = ""
    for new_text in streamer:
      buffer += new_text
      yield buffer


with gr.Blocks() as demo:
  with gr.Row():
    with gr.Column():
      img_input = gr.Image(label="camera", sources="webcam", streaming=True, type="pil")
    with gr.Column():
      txt_input = gr.Textbox(label="Ask a question", lines=2)
      btn_input = gr.Button("Submit")
      txt_output = gr.Textbox(label="LLM Anwsers", lines=2)

  img_input.stream(fn_camera, img_input, img_input, stream_every=0.5, concurrency_limit=30)
  txt_input.submit(fn_llm, [img_input, txt_input], txt_output, queue=True)
  btn_input.click(fn_llm, [img_input, txt_input], txt_output, queue=True)

if __name__ == "__main__":
  demo.launch()
