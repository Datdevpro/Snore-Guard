import tensorflow as tf

def convert_and_optimize_tflite(model_filename, tflite_filename, optimized_tflite_filename):
    # Load the Keras model
    model = tf.keras.models.load_model(model_filename)

    # Convert the model to TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(tflite_filename, 'wb') as f:
        f.write(tflite_model)
    print(f"Model converted to {tflite_filename}")

    # Optimize the TensorFlow Lite model
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    optimized_tflite_model = converter.convert()
    with open(optimized_tflite_filename, 'wb') as f:
        f.write(optimized_tflite_model)
    print(f"Model optimized and saved to {optimized_tflite_filename}")

if __name__ == "__main__":
    model_filename = 'model/snoring_detector_model_0.99_2.h5'
    tflite_filename = 'model/snoring_detector_model.tflite'
    optimized_tflite_filename = 'model/snoring_detector_model_optimized.tflite'
    convert_and_optimize_tflite(model_filename, tflite_filename, optimized_tflite_filename)
