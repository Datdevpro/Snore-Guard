import librosa
import numpy as np

def extract_features(audio_filename):
    y, sr = librosa.load(audio_filename, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)
