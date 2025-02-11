import argparse
import sys

def is_venv():
  return (hasattr(sys, 'real_prefix') or
    (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))

def pip_install(*args):
  import subprocess  # nosec - disable B404:import-subprocess check

  cli_args = []
  for arg in args:
    cli_args.extend(str(arg).split(" "))
  subprocess.run([sys.executable, "-m", "pip", "install", *cli_args], check=True)

def pip_uninstall(*args):
  import subprocess  # nosec - disable B404:import-subprocess check

  cli_args = []
  for arg in args:
    cli_args.extend(str(arg).split(" "))
  subprocess.run([sys.executable, "-m", "pip", "uninstall", *cli_args], check=True)

def download_model(nightly):
  pip_install("-Uq", "pip")
  pip_install("gradio>=4.19", "python-dotenv", "transformers", "intel-npu-acceleration-library")

  if nightly == "true":
    pip_uninstall("openvino", "openvino-tokenizers", "openvino_genai")
    pip_install("--pre", "--extra-index-url", "https://storage.openvinotoolkit.org/simple/wheels/nightly", "openvino", "openvino-tokenizers", "openvino_genai")
  else:
    pip_uninstall("openvino", "openvino-tokenizers", "openvino_genai")
    pip_install("openvino", "openvino-tokenizers", "openvino_genai")



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "-n",
    "--nightly",
    default="false",
    help="Nightly flag for downloading openvino nightly builds",
  )
  args = parser.parse_args()

  if is_venv():
    download_model(args.nightly)
  else:
    print("Not running inside a virtual environment")
