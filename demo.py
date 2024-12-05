from modelsOV import pipe
from demo_helper import make_demo
from modelsConfig import model_configuration

demo = make_demo(pipe, model_configuration, model_configuration["model_id"], "English")

try:
    demo.launch(debug=True)
except Exception:
    demo.launch(debug=True, share=True)
