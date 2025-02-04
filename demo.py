def demo():
  from llm import pipe
  from demo_helper import make_demo
  from llm_config import model_configuration

  demo = make_demo(pipe, model_configuration, model_configuration["model_id"], "English")

  try:
    demo.launch(debug=True)
  except Exception:
    demo.launch(debug=True, share=True)

if __name__ == "__main__":
  demo()
