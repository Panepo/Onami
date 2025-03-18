def demo():
  import copy
  import numpy as np
  from noise_suppression import compiled_model, wav_read, wav_write, inp_shapes, out_shapes, state_inp_names

  infer_request = compiled_model.create_infer_request()

  sample_inp, freq_data = wav_read(str("./courtroom.wav"))
  sample_size = sample_inp.shape[0]

  infer_request.infer()
  delay = 0
  if "delay" in out_shapes:
    delay = infer_request.get_tensor("delay").data[0]
    sample_inp = np.pad(sample_inp, ((0, delay), ))
  freq_model = 16000
  if "freq" in out_shapes:
    freq_model = infer_request.get_tensor("freq").data[0]

  if freq_data != freq_model:
    raise RuntimeError(
      "Wav file {} sampling rate {} does not match model sampling rate {}".
          format("./courtroom.wav", freq_data, freq_model))

  input_size = inp_shapes["input"][1]
  res = None

  samples_out = []
  while sample_inp is not None and sample_inp.shape[0] > 0:
      if sample_inp.shape[0] > input_size:
          input = sample_inp[:input_size]
          sample_inp = sample_inp[input_size:]
      else:
          input = np.pad(sample_inp, ((0, input_size - sample_inp.shape[0]), ), mode='constant')
          sample_inp = None

      #forms input
      inputs = {"input": input[None, :]}

      #add states to input
      for n in state_inp_names:
          if res:
              inputs[n] = infer_request.get_tensor(n.replace('inp', 'out')).data
          else:
              #on the first iteration fill states by zeros
              inputs[n] = np.zeros(inp_shapes[n], dtype=np.float32)

      infer_request.infer(inputs)
      res = infer_request.get_tensor("output")
      samples_out.append(copy.deepcopy(res.data).squeeze(0))

  sample_out = np.concatenate(samples_out, 0)
  sample_out = sample_out[delay:sample_size+delay]
  wav_write("./output.wav", sample_out, freq_data)


if __name__ == "__main__":
  demo()
