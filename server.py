from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
  print("Chat request")
  print("================================================")
  data = request.get_json()
  system = data.get('system', '')
  prompt = data.get('prompt', '')
  print(f"System: {system}")
  print(f"Prompt: {prompt}")

  if "completion_to_prompt" in model_configuration:
    completion_to_prompt = model_configuration["completion_to_prompt"]
    prompt = completion_to_prompt(system, prompt)

  response = pipe.generate(prompt, config)
  print(f"Response:\n{response}")
  print("================================================")
  return response

if __name__ == '__main__':
  from llm import pipe, config
  from llm_config import model_configuration
  app.run(host='0.0.0.0', port=8090)
