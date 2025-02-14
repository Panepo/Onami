import argparse
from llm_config import model_dir, model_path

def download_model(model):
  if not model_dir.exists():
    model_dir.mkdir()

  def git_clone(repo, path):
    import subprocess  # nosec - disable B404:import-subprocess check
    subprocess.run(["git", "clone", repo, path], check=True)

  target_dir = model_dir / model_path[model]

  if not target_dir.exists() and model == "llama3.1":
    print("Downloading Llama-3.1 8B model...")
    git_clone("https://huggingface.co/AIFunOver/Llama-3.1-8B-Instruct-openvino-4bit", target_dir)

  if not target_dir.exists() and model == "llama3.1draft":
    print("Downloading Llama-3.1 8B fastdraft model...")
    git_clone("https://huggingface.co/OpenVINO/Llama-3.1-8B-Instruct-FastDraft-150M-int8-ov", target_dir)

  if not target_dir.exists() and model == "llama3.2":
    print("Downloading Llama-3.2 3B model...")
    git_clone("https://huggingface.co/AIFunOver/Llama-3.2-3B-Instruct-openvino-4bit", target_dir)

  if not target_dir.exists() and model == "tinyllama":
    print("Downloading TinyLlama 1.5B model...")
    git_clone("https://huggingface.co/OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov", target_dir)

  if not target_dir.exists() and model == "phi3":
    print("Downloading Phi-3 4B model...")
    git_clone("https://huggingface.co/OpenVINO/Phi-3-mini-4k-instruct-int4-ov", target_dir)

  if not target_dir.exists() and model == "phi3.5":
    print("Downloading Phi-3.5 4B model...")
    git_clone("https://huggingface.co/OpenVINO/Phi-3.5-mini-instruct-int4-ov", target_dir)

  if not target_dir.exists() and model == "phi4":
    print("Downloading Phi-4 14B model...")
    git_clone("https://huggingface.co/AIFunOver/phi-4-openvino-4bit", target_dir)

  if not target_dir.exists() and model == "gemma2":
    print("Downloading Gemma-2 2B model...")
    git_clone("https://huggingface.co/OpenVINO/gemma-2b-it-int4-ov", target_dir)

  if not target_dir.exists() and model == "gemma29":
    print("Downloading Gemma-2 9B model...")
    git_clone("https://huggingface.co/OpenVINO/gemma-2-9b-it-int4-ov", target_dir)

  if not target_dir.exists() and model == "phi3.5vision":
    print("Downloading Phi-3.5 vision model...")
    git_clone("https://huggingface.co/OpenVINO/Phi-3.5-vision-instruct-int4-ov", target_dir)

  if not target_dir.exists() and model == "internvl2":
    print("Downloading InternVL2 2B model...")
    git_clone("https://huggingface.co/OpenVINO/InternVL2-2B-int4-ov", target_dir)


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
