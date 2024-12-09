from dotenv import load_dotenv
import os

load_dotenv()
nightly = os.getenv("NIGHTLY")

import sys

def pip_install(*args):
  import subprocess  # nosec - disable B404:import-subprocess check

  cli_args = []
  for arg in args:
    cli_args.extend(str(arg).split(" "))
  subprocess.run([sys.executable, "-m", "pip", "install", *cli_args], check=True)

pip_install("-Uq", "pip")
pip_install("gradio>=4.19", "python-dotenv")

if nightly == "true":
  pip_install("--pre", "--extra-index-url", "https://storage.openvinotoolkit.org/simple/wheels/nightly", "openvino", "openvino-tokenizers", "openvino_genai")
else:
  pip_install("openvino", "openvino-tokenizers", "openvino_genai")
