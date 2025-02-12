import tensorflow as tf

def convert_to_tflite(model_filename, tflite_filename):
    model = tf.keras.models.load_model(model_filename)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(tflite_filename, 'wb') as f:
        f.write(tflite_model)
    print(f"Model converted to {tflite_filename}")

if __name__ == "__main__":
    model_filename = 'model/snoring_detector_model_0.99_2.h5'
    tflite_filename = 'model/snoring_detector_model2.tflite'
    convert_to_tflite(model_filename, tflite_filename)
