from pathlib import Path

model_dir = Path("models")

llama32_dir = model_dir / "Llama-3.2-3B-Instruct-openvino-4bit"
tinyllama_dir = model_dir / "TinyLlama-1.1B-Chat-v1.0-int4-ov"
phi3_dir = model_dir / "Phi-3-mini-4k-instruct-int4-ov"
phi35_dir = model_dir / "Phi-3.5-4k-instruct-int4-ov"

def download_model():
  if not model_dir.exists():
    model_dir.mkdir()

  def git_clone(repo, path):
    import subprocess  # nosec - disable B404:import-subprocess check
    subprocess.run(["git", "clone", repo, path], check=True)

  if not llama32_dir.exists():
    git_clone("https://huggingface.co/AIFunOver/Llama-3.2-3B-Instruct-openvino-4bit", llama32_dir)

  if not tinyllama_dir.exists():
    git_clone("https://huggingface.co/OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov", tinyllama_dir)

  if not phi3_dir.exists():
    git_clone("https://huggingface.co/OpenVINO/Phi-3-mini-4k-instruct-int4-ov", phi3_dir)

  if not phi35_dir.exists():
    git_clone("https://huggingface.co/OpenVINO/Phi-3.5-4k-instruct-int4-ov", phi35_dir)

if __name__ == "__main__":
  download_model()