import argparse

def demo_chat():
  def streamer(subword):
    print(subword, end='', flush=True)
    return False

  start_message = model_configuration.get("start_message", DEFAULT_SYSTEM_PROMPT)
  #pipe.start_chat(system_message=start_message)
  try:
    while 1:
      pipe.start_chat(system_message=start_message)
      print("================================================")
      message = input("Please say something: ")
      if message == "exit" or message == "quit" or message == "":
        pipe.finish_chat()
        break

      if "completion_to_prompt" in model_configuration:
        completion_to_prompt = model_configuration["completion_to_prompt"]
        message = completion_to_prompt(message)

      print("Assistant: ")
      pipe.generate(message, config, streamer)
      print("\n")
      pipe.finish_chat()
  except KeyboardInterrupt:
    pipe.finish_chat()

def demo_perf():
  try:
    while 1:
      print("================================================")
      message = input("Please say something: ")
      if message == "exit" or message == "quit" or message == "":
        break

      if "completion_to_prompt" in model_configuration:
        completion_to_prompt = model_configuration["completion_to_prompt"]
        message = completion_to_prompt(message)

      response = pipe.generate([message], config)

      print("================================================")
      print(f"Assistant: {response}")

      print("================================================")
      perf_metrics = response.perf_metrics
      print(f"Load time: {perf_metrics.get_load_time():.2f} ms")
      print(f"Input tokens: {perf_metrics.get_num_input_tokens()}")
      print(f"Generated tokens: {perf_metrics.get_num_generated_tokens()}")
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
  from llm_config import model_configuration, DEFAULT_SYSTEM_PROMPT

  if args.chat:
    print("Current running on Chat streaming mode")
    demo_chat()
  else:
    print("Current running on Performance checking mode")
    demo_perf()
