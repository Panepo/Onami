import time
start_time = time.process_time()

from modelsOV import pipe
from demo_helper import make_demo
from modelsConfig import model_configuration

demo = make_demo(pipe, model_configuration, model_configuration["model_id"], "English")

try:
    end_time = time.process_time()
    elapsed_time = end_time - start_time
    print(f"Load time: {elapsed_time} seconds")
    demo.launch(debug=True)
except Exception:
    demo.launch(debug=True, share=True)
