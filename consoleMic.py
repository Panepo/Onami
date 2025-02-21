def demo():
  import pyaudio, wave
  import numpy as np
  from noise_suppression import compiled_model

  infer_request = compiled_model.create_infer_request()

  # Audio stream setup
  CHUNK = 128
  RATE = 16000
  p = pyaudio.PyAudio()

  # Open a .wav file to save the output
  wf = wave.open("output.wav", 'wb')
  wf.setnchannels(1)
  wf.setsampwidth(p.get_sample_size(pyaudio.paFloat32))
  wf.setframerate(RATE)

  def callback(in_data, frame_count, time_info, status):
    audio_data = np.frombuffer(in_data, dtype=np.float32)
    audio_data = audio_data.reshape(1, 128)
    # Perform noise suppression inference
    infer_request.infer(inputs={0: audio_data})
    output_data = infer_request.get_output_tensor(0).data
    wf.writeframes(output_data.tobytes())
    return (output_data.tobytes(), pyaudio.paContinue)

  stream = p.open(format=pyaudio.paFloat32,
                  channels=1,
                  rate=RATE,
                  input=True,
                  output=True,
                  frames_per_buffer=CHUNK,
                  stream_callback=callback)

  print("Noise suppression is running. Press Ctrl+C to stop.")
  try:
    stream.start_stream()
    while stream.is_active():
      pass
  except KeyboardInterrupt:
    pass
  finally:
    stream.stop_stream()
    stream.close()
    wf.close()
    p.terminate()

if __name__ == "__main__":
  demo()
