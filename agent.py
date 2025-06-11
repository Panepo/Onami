import json
import openvino_genai as ov_genai

class ToolParameter:
  def __init__(self, name: str, description: str, type: str, default=None, required=False) -> None:
    self.name = name
    self.description = description
    self.type = type
    self.default = default
    self.required = required

class Tool:
  def __init__(self, name: str) -> None:
    self.name = name
    self.description = ""
    self.parameters = []

  def set_name(self, name: str) -> None:
    self.name = name

  def set_description(self, description: str) -> None:
    self.description = description

  def add_parameter(self, parameter: ToolParameter) -> None:
    self.parameters.append(parameter)

  def get_tool(self) -> json:
    return {
      "name": self.name,
      "description": self.description,
      "parameters": {
        "type": "dict",
        "required": [parameter.name for parameter in self.parameters if parameter.required],
        "properties": {
          parameter.name: {
            "type": parameter.type,
            "description": parameter.description,
            "default": parameter.default
          } for parameter in self.parameters
        }
      }
    }

DEFAULT_SYSTEM_PROMPT_JSON = """\
You are an expert in composing functions. You are given a question and a set of possible functions.
Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out. You should only return the function call in tools call sections.
If you decide to invoke any of the function(s), you MUST respond in the format {"name": function name1, "parameters": dictionary of argument name and its value}, {"name": function name2, "parameters": dictionary of argument name and its value}. Do not use variables.
You SHOULD NOT include any other text in the response. Here is a list of functions in JSON format that you can invoke.
"""

DEFAULT_SYSTEM_PROMPT_ARRAY = """\
You are an expert in composing functions. You are given a question and a set of possible functions.
Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out. You should only return the function call in tools call sections.
If you decide to invoke any of the function(s), you MUST respond in the format [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]. Do not use variables.
You SHOULD NOT include any other text in the response. Here is a list of functions in JSON format that you can invoke.
"""

DEFAULT_SYSTEM_PROMPT_END = """"""

PROMPT_HEADER_START = "<|start_header_id|>system<|end_header_id|>"
PROMPT_HEADER_END = "<|eot_id|><|start_header_id|>user<|end_header_id|>"
PROMPT_FOOTER = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

class Agent:
  def __init__(self, llm: ov_genai.LLMPipeline = None, tools: list[Tool] = []) -> None:
    self.llm = llm
    self.tools = tools

  def set_llm(self, llm: ov_genai.LLMPipeline) -> None:
    self.llm = llm

  def add_tool(self, tool: Tool) -> None:
    self.tools.append(tool)

  def get_system_prompt(self) -> str:
    return DEFAULT_SYSTEM_PROMPT_ARRAY + "[" + ", ".join([json.dumps(tool.get_tool()) for tool in self.tools]) + "]" + DEFAULT_SYSTEM_PROMPT_END

  def completion_to_prompt(self, system: str, message: str) -> str:
    return PROMPT_HEADER_START + system + PROMPT_HEADER_END + message + PROMPT_FOOTER

  def generate(self, message: str, config: ov_genai.GenerationConfig) -> str:
    if self.llm is None:
      raise ValueError("LLM model not initialized")

    prompt = self.completion_to_prompt(self.get_system_prompt(), message)
    return self.llm.generate(prompt, config)
