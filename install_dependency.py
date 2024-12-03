import sys

def pip_install(*args):
    import subprocess  # nosec - disable B404:import-subprocess check

    cli_args = []
    for arg in args:
        cli_args.extend(str(arg).split(" "))
    subprocess.run([sys.executable, "-m", "pip", "install", *cli_args], check=True)

pip_install("-Uq", "pip")
pip_install("-q", "openvino>=2024.3.0", "openvino-tokenizers[transformers]", "openvino-genai")
pip_install("-q", "--extra-index-url", "https://download.pytorch.org/whl/cpu",
            "git+https://github.com/huggingface/optimum-intel.git",
            "git+https://github.com/openvinotoolkit/nncf.git",
            "torch>=2.1", "datasets", "accelerate", "gradio>=4.19", "transformers>=4.43.1",
            "onnx<=1.16.1", "einops", "transformers_stream_generator",
            "tiktoken", "bitsandbytes")

