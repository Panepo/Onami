import argparse

def demo(prompt:str, negative_prompt: str, input_image_path:str):
  import openvino as ov
  import numpy as np
  from PIL import Image
  import openvino_genai as ov_genai
  from img_gen import pipe

  def image_to_tensor(image: Image) -> ov.Tensor:
    pic = image.convert("RGB")
    image_data = np.array(pic.getdata()).reshape(1, pic.size[1], pic.size[0], 3).astype(np.uint8)
    return ov.Tensor(image_data)

  input_image = Image.open(input_image_path)
  input_image_tensor = image_to_tensor(input_image)

  random_generator = ov_genai.TorchGenerator(42)

  image_tensor = pipe.generate(
    prompt,
    input_image_tensor,
    negative_prompt=negative_prompt,
    strength=0.75,
    num_inference_steps=20,
    num_images_per_prompt=1,
    generator=random_generator,
  )

  out_image = Image.fromarray(image_tensor.data[0])
  out_image.save("output.png")
  out_image.show()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "-i",
    "--input",
    required=True,
    help="Path to the input image",
  )
  parser.add_argument(
    "-p",
    "--prompt",
    required=True,
    help="Prompt for the model",
  )
  parser.add_argument(
    "-n",
    "--negative",
    default="",
    help="Negative prompt for the model",
  )
  args = parser.parse_args()

  demo(args.prompt, args.negative, args.input)
