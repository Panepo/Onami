import pyaudio

if __name__ == "__main__":
  p = pyaudio.PyAudio()

  # List all available audio input devices
  for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info["maxInputChannels"] > 0:
      print(f"Device ID: {i}, Name: {info['name']}")
