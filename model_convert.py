import argparse
import subprocess  # nosec - disable B404:import-subprocess check
import sys
from pathlib import Path
from typing import Dict
import platform
import logging

def optimum_cli(model_id, output_dir, show_command=True, additional_args: Dict[str, str] = None):
  export_command = f"optimum-cli export openvino --model {model_id} {output_dir}"
  if additional_args is not None:
    for arg, value in additional_args.items():
      export_command += f" --{arg}"
      if value:
        export_command += f" {value}"

  if show_command:
    print(f"`{export_command}`")

  try:
    subprocess.run(export_command.split(" "), shell=(platform.system() == "Windows"), check=True, capture_output=True)
  except subprocess.CalledProcessError as exc:
    logger = logging.getLogger()
    logger.exception(exc.stderr)
    raise exc
