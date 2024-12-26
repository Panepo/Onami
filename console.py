import argparse

def demo_chat():
  def streamer(subword):
    print(subword, end='', flush=True)
    return False

  pipe.start_chat()
  try:
    while 1:
      print("================================================")
      message = input("Please say something: ")
      if message == "exit" or message == "quit" or message == "":
        pipe.finish_chat()
        break

      start_message = model_configuration.get("start_message", DEFAULT_SYSTEM_PROMPT)
      console_message_start = model_configuration.get("console_message_start", "")
      console_message_end = model_configuration.get("console_message_end", "")
      print("Assistant: ")
      pipe.generate(start_message + console_message_start + message + console_message_end, config, streamer)
      print("\n")
  except KeyboardInterrupt:
    pipe.finish_chat()

def demo_perf():
  try:
    while 1:
      print("================================================")
      message = input("Please say something: ")
      if message == "exit" or message == "quit" or message == "":
        break

      start_message = model_configuration.get("start_message", DEFAULT_SYSTEM_PROMPT)
      console_message_start = model_configuration.get("console_message_start", "")
      console_message_end = model_configuration.get("console_message_end", "")
      response = pipe.generate([start_message + console_message_start + message + console_message_end], config)

      print("================================================")
      print(f"Assistant: {response}")

      print("================================================")
      perf_metrics = response.perf_metrics
      print(f"Load time: {perf_metrics.get_load_time():.2f} ms")
      print(f"Generate time: {perf_metrics.get_generate_duration().mean:.2f} ± {perf_metrics.get_generate_duration().std:.2f} ms")
      print(f"Tokenization time: {perf_metrics.get_tokenization_duration().mean:.2f} ± {perf_metrics.get_tokenization_duration().std:.2f} ms")
      print(f"Detokenization time: {perf_metrics.get_detokenization_duration().mean:.2f} ± {perf_metrics.get_detokenization_duration().std:.2f} ms")
      print(f"TTFT: {perf_metrics.get_ttft().mean:.2f} ± {perf_metrics.get_ttft().std:.2f} ms")
      print(f"TPOT: {perf_metrics.get_tpot().mean:.2f} ± {perf_metrics.get_tpot().std:.2f} ms")
      print(f"Throughput : {perf_metrics.get_throughput().mean:.2f} ± {perf_metrics.get_throughput().std:.2f} tokens/s")
  except KeyboardInterrupt:
    pass

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "-c",
    "--chat",
    default=False,
    help="(optional) Enable chat streaming mode",
  )
  args = parser.parse_args()

  from llm import pipe, config
  from model_config import model_configuration, DEFAULT_SYSTEM_PROMPT

  if args.chat:
    print("Current running on Chat streaming mode")
    demo_chat()
  else:
    print("Current running on Performance checking mode")
    demo_perf()
