import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization

def create_model(input_shape):
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(128, input_shape=(input_shape[0], input_shape[1]), return_sequences=True),
        BatchNormalization(),
        Dropout(0.4),
        LSTM(64, return_sequences=False),
        BatchNormalization(),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def summarize_model(model):
    model.summary()
    model_size = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    print(f'Model size: {model_size} parameters')
