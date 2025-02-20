from load_dataset import load_dataset
from feature_extraction import extract_features
from model import create_lstm_model, summarize_model
from evaluation import plot_history, plot_confusion_matrix, get_model_size 
from test_model import test_model
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping

# Load dataset
snoring_dir = '/home/lee/work/CP/magic_pillow/dataset/snoring'
no_snoring_dir = '/home/lee/work/CP/magic_pillow/dataset/no_snoring'
X, y = load_dataset(snoring_dir, no_snoring_dir)

# Reshape data for LSTM
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = create_lstm_model(input_shape=(X_train.shape[1], 1))
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, callbacks=[early_stopping])

# Evaluate model
plot_history(history)
y_pred = model.predict(X_test)
plot_confusion_matrix(y_test, y_pred > 0.5)

# Show ground truth table
#show_ground_truth_table(y_test, y_pred)

# Calculate accuracy
accuracy = np.mean((y_pred > 0.5).flatten() == y_test)
loss, acc = model.evaluate(X_test, y_test, verbose=2)

# Show classification report
y_pred_classes = (y_pred > 0.5).astype(int)
print(classification_report(y_test, y_pred_classes))

# Save model with accuracy in filename
model_filename = f'snoring_detector_lstm_model_{accuracy:.2f}.h5'
model.save(model_filename)

# Summarize model
print(f'Model accuracy: {accuracy * 100:.2f}%')
print(f'Model test set accuracy: {acc * 100:.2f}%')
summarize_model(model)
get_model_size(model_filename)

# Test model with user-provided audio file
result = test_model(model_filename, 'snoring_testing_out.wav')
print(result)


