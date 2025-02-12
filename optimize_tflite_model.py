import tensorflow as tf

def optimize_tflite_model(tflite_filename, optimized_tflite_filename):
    # Load the TensorFlow Lite model
    with open(tflite_filename, 'rb') as f:
        tflite_model = f.read()

    # Convert the model to a TensorFlow Lite model with optimization
    converter = tf.lite.TFLiteConverter.from_keras_model(tflite_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    optimized_tflite_model = converter.convert()

    # Save the optimized model
    with open(optimized_tflite_filename, 'wb') as f:
        f.write(optimized_tflite_model)
    print(f"Model optimized and saved to {optimized_tflite_filename}")

if __name__ == "__main__":
    tflite_filename = 'model/snoring_detector_model.tflite'
    optimized_tflite_filename = 'model/snoring_detector_model_optimized.tflite'
    optimize_tflite_model(tflite_filename, optimized_tflite_filename)
