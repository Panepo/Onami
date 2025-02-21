def demo():
  import librosa
  from whisper import pipe, config

  en_raw_speech, samplerate = librosa.load(str("./courtroom.wav"), sr=16000)

  genai_result = pipe.generate(en_raw_speech, config)
  print(f"Result: {genai_result}")

if __name__ == "__main__":
  demo()
