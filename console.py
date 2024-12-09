import time

prompt = ""

def demo():
  from llm import pipe, config, model_dir

  while 1:
    print("================================================")
    message = input("Please say something: ")
    response = pipe.generate([message], config)
    print(response)

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
