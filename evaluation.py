import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import tensorflow as tf
import pandas as pd

def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig("training_plot_2.png")  

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig("cfs_matrix_2.png")  

def get_model_size(model_path=str):
    """Calculates and prints the size of a saved Keras model in MB and number of trainable parameters."""
    # Get file size in MB
    model_size = os.path.getsize(model_path) / (1024 * 1024)  # Convert bytes to MB
    
    # Load the model
    model = tf.keras.models.load_model(model_path)

    # Get total number of trainable parameters
    total_params = model.count_params()

    print(f"📦 Model Size: {model_size:.2f} MB")
    print(f"⚙️ Trainable Parameters: {total_params:,}")

    return model_size, total_params

