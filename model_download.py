import argparse
from llm_config import model_dir, llama32_dir, tinyllama_dir, phi3_dir, phi35_dir, llama3_dir

def download_model(model):
  if not model_dir.exists():
    model_dir.mkdir()

  def git_clone(repo, path):
    import subprocess  # nosec - disable B404:import-subprocess check
    subprocess.run(["git", "clone", repo, path], check=True)

  if not llama32_dir.exists() and model == "llama3.2":
    print("Downloading Llama-3.2 3B model...")
    git_clone("https://huggingface.co/AIFunOver/Llama-3.2-3B-Instruct-openvino-4bit", llama32_dir)

  if not tinyllama_dir.exists() and model == "tinyllama":
    print("Downloading TinyLlama 1.5B model...")
    git_clone("https://huggingface.co/OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov", tinyllama_dir)

  if not phi3_dir.exists() and model == "phi3":
    print("Downloading Phi-3 4B model...")
    git_clone("https://huggingface.co/OpenVINO/Phi-3-mini-4k-instruct-int4-ov", phi3_dir)

  if not phi35_dir.exists() and model == "phi3.5":
    print("Downloading Phi-3.5 4B model...")
    git_clone("https://huggingface.co/OpenVINO/Phi-3.5-mini-instruct-int4-ov", phi35_dir)

  if not llama3_dir.exists() and model == "llama3.1":
    print("Downloading Llama-3.1 8B model...")
    git_clone("https://huggingface.co/AIFunOver/Llama-3.1-8B-Instruct-openvino-4bit", llama3_dir)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "-m",
    "--model",
    required=True,
    help="Download model, currently supported models are llama3.2, tinyllama, phi3, phi3.5, llama3.1",
  )
  args = parser.parse_args()

  download_model(args.model)
