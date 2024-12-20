def demo():
  from llm import pipe, config
  from model_config import model_configuration, DEFAULT_SYSTEM_PROMPT

  while 1:
    print("================================================")
    message = input("Please say something: ")
    if message == "exit":
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

if __name__ == "__main__":
  demo()
