from pathlib import Path

def banchmark():
  from vlm import pipe, config, read_image_path

  image_url = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11"
  image_file = Path("cat.png")

  if not image_file.exists():
    image, image_tensor = read_image_path(image_url)
    image.save(image_file)
  else:
    image, image_tensor = read_image_path(image_file)

  text_message = "What is unusual on this image?"

  prompt = text_message
  output = pipe.generate(prompt, image=image_tensor, generation_config=config)
  print(output)
  perf_metrics = output.perf_metrics

  print("\n================================================")
  print(f"Load time: {perf_metrics.get_load_time():.2f} ms")
  print(f"Generate time: {perf_metrics.get_generate_duration().mean:.2f} ± {perf_metrics.get_generate_duration().std:.2f} ms")
  print(f"Tokenization time: {perf_metrics.get_tokenization_duration().mean:.2f} ± {perf_metrics.get_tokenization_duration().std:.2f} ms")
  print(f"Detokenization time: {perf_metrics.get_detokenization_duration().mean:.2f} ± {perf_metrics.get_detokenization_duration().std:.2f} ms")
  print(f"TTFT: {perf_metrics.get_ttft().mean:.2f} ± {perf_metrics.get_ttft().std:.2f} ms")
  print(f"TPOT: {perf_metrics.get_tpot().mean:.2f} ± {perf_metrics.get_tpot().std:.2f} ms")
  print(f"Throughput : {perf_metrics.get_throughput().mean:.2f} ± {perf_metrics.get_throughput().std:.2f} tokens/s")

if __name__ == "__main__":
  banchmark()
