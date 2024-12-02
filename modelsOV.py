from transformers import AutoConfig, AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM

import openvino as ov
import openvino.properties as props
import openvino.properties.hint as hints
import openvino.properties.streams as streams
import torch
from transformers import (
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

model_dir = "models/Llama-3.2-3B-Instruct-ov-int4"
ov_config = {hints.performance_mode(): hints.PerformanceMode.LATENCY, streams.num(): "1", props.cache_dir(): ""}

# On a GPU device a model is executed in FP16 precision. For red-pajama-3b-chat model there known accuracy
# issues caused by this, which we avoid by setting precision hint to "f32".
core = ov.Core()

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

tokenizer_kwargs = {}

class StopOnTokens(StoppingCriteria):
  def __init__(self, token_ids):
    self.token_ids = token_ids

  def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
    for stop_id in self.token_ids:
      if input_ids[0][-1] == stop_id:
        return True
    return False

stop_tokens = ["<|eot_id|>"]
if stop_tokens is not None:
  if isinstance(stop_tokens[0], str):
    stop_tokens = tokenizer.convert_tokens_to_ids(stop_tokens)

  stop_tokens = [StopOnTokens(stop_tokens)]

generate_kwargs = dict(
  max_length=2048,
  max_new_tokens=2048,
  temperature=0.7,
  do_sample=True,
  top_p=0.95,
  top_k=50,
  stopping_criteria=StoppingCriteriaList(stop_tokens),
)

ovmodel = OVModelForCausalLM.from_pretrained(
  model_dir,
  device="NPU",
  ov_config=ov_config,
  config=AutoConfig.from_pretrained(model_dir, trust_remote_code=True),
  trust_remote_code=True,
  generate_kwargs=generate_kwargs
)

import time

test_string = """
You are a helpful assistant that identify the person, time, place and what happen in brief in the following article.
On Friday, January 1, 2016, at approximately 8:15 AM, I parked my car in the Food Mart parking lot, located at 1234 Food Dr., in Goleta. I left my car unlocked since I was planning to return quickly. While I was inside the Food Mart, I had realized I left my cellphone on top of the center console of my car. When I returned to my car, at approximately 8:20 AM, I discovered my car’s front passenger door was ajar and my cellphone was gone. I did not see who stole my cellphone or know who could have stolen it. This report is for documentation purposes only.
"""
start_time = time.process_time()
input_tokens = tokenizer(test_string, return_tensors="pt", **tokenizer_kwargs)
answer = ovmodel.generate(**input_tokens, **generate_kwargs)
print(tokenizer.batch_decode(answer, skip_special_tokens=True)[0])
end_time = time.process_time()
elapsed_time = end_time - start_time
print(f"Process time: {elapsed_time} seconds")
