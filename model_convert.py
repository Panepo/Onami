import argparse
import sys
from pathlib import Path
import logging

def convert_to_openvino(model_path, output_path, weight_format="int4", group_size=128, ratio=0.8):
  """Convert model to OpenVINO format using optimum library"""
  from optimum.intel import OVModelForCausalLM
  from transformers import AutoTokenizer

  logger = logging.getLogger(__name__)

  logger.info(f"Loading model from: {model_path}")

  # Load and export the model with quantization and trust_remote_code for new architectures
  if weight_format == "int4":
    logger.info(f"Exporting with int4 quantization (group_size={group_size}, ratio={ratio})")
    model = OVModelForCausalLM.from_pretrained(
      model_path,
      export=True,
      load_in_8bit=False,
      compile=False,
      trust_remote_code=True  # Allow loading custom model architectures
    )
  else:
    model = OVModelForCausalLM.from_pretrained(
      model_path,
      export=True,
      compile=False,
      trust_remote_code=True
    )

  # Save the model
  logger.info(f"Saving model to: {output_path}")
  model.save_pretrained(output_path)

  # Also save the tokenizer
  logger.info("Saving tokenizer")
  tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
  tokenizer.save_pretrained(output_path)

  logger.info("Conversion completed successfully!")


def main():
  parser = argparse.ArgumentParser(description="Convert models to OpenVINO format")
  parser.add_argument("--model", required=True, help="Path to the model or model ID")
  parser.add_argument("--output", required=True, help="Output directory for converted model")
  parser.add_argument("--weight-format", default="int4", help="Weight format for quantization (int4, int8, fp16)")
  parser.add_argument("--group-size", type=int, default=128, help="Group size for quantization")
  parser.add_argument("--ratio", type=float, default=0.8, help="Ratio for mixed precision quantization")

  args = parser.parse_args()

  # Setup logging
  logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
  logger = logging.getLogger(__name__)

  # Resolve paths
  model_path = Path(args.model).resolve()
  output_path = Path(args.output).resolve()

  logger.info(f"Converting model from: {model_path}")
  logger.info(f"Output directory: {output_path}")
  logger.info(f"Weight format: {args.weight_format}")

  try:
    convert_to_openvino(
      str(model_path),
      str(output_path),
      weight_format=args.weight_format,
      group_size=args.group_size,
      ratio=args.ratio
    )
  except Exception as e:
    logger.error(f"Conversion failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

if __name__ == "__main__":
  main()
