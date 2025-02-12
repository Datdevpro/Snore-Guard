import numpy as np
from feature_extraction import extract_features
from tensorflow.keras.models import load_model

def test_model(model_filename, audio_filename):
    model = load_model(model_filename)
    features = extract_features(audio_filename)
    features = np.expand_dims(features, axis=0)
    prediction = model.predict(features)
    return "Snoring Detected 💤 " if prediction[0][0] > 0.5 else "No Snoring Detected ❌"
