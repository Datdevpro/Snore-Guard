import pyaudio
import wave
import numpy as np
from feature_extraction import extract_features
from tensorflow.keras.models import load_model
import time

def record_sound(duration=5, sample_rate=44100, channels=1):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=channels, rate=sample_rate, input=True, frames_per_buffer=1024)
    frames = []

    print("Recording...")
    for _ in range(0, int(sample_rate / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)

    print("Recording finished.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open('real_time_audio.wav', 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()

def real_time_detection(model_filename):
    model = load_model(model_filename)
    while True:
        record_sound(duration=5)
        features = extract_features('real_time_audio.wav')
        features = np.expand_dims(features, axis=0)
        prediction = model.predict(features)
        result = "Snoring Detected 💤 " if prediction[0][0] > 0.5 else "No Snoring Detected ❌"
        print(result)
        time.sleep(1)  # Wait for 1 second before recording the next audio

if __name__ == "__main__":
    model_filename = 'model/snoring_detector_model_0.99_2.h5'
    real_time_detection(model_filename)
