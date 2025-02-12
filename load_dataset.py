import os
import numpy as np
from feature_extraction import extract_features

def load_dataset(snoring_dir, no_snoring_dir):
    X = []
    y = []

    for filename in os.listdir(snoring_dir):
        if filename.endswith('.wav'):
            filepath = os.path.join(snoring_dir, filename)
            features = extract_features(filepath)
            X.append(features)
            y.append(1)  # Snoring label

    for filename in os.listdir(no_snoring_dir):
        if filename.endswith('.wav'):
            filepath = os.path.join(no_snoring_dir, filename)
            features = extract_features(filepath)
            X.append(features)
            y.append(0)  # No snoring label

    return np.array(X), np.array(y)
