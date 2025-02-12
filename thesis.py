import os
from scipy.io import wavfile
import numpy as np
from sklearn.preprocessing import StandardScaler
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
import matplotlib.pyplot as plt

def read_wav_files(directory):
    wav_files = {}
    try:
        for filename in os.listdir(directory):
            if filename.endswith('.wav'):
                filepath = os.path.join(directory, filename)
                try:
                    sample_rate, data = wavfile.read(filepath)
                    wav_files[filename] = {'sample_rate': sample_rate, 'data': data}
                except Exception as e:
                    print(f"Error reading WAV file {filename}: {e}")
    except FileNotFoundError:
        print(f"Directory not found: {directory}")
    return wav_files

def extract_tonnetz_from_wav(directory):
    tonnetz_list = []
    try:
        for filename in os.listdir(directory):
            if filename.endswith('.wav'):
                filepath = os.path.join(directory, filename)
                try:
                    data, sample_rate = librosa.load(filepath)
                    tonnetz = np.array(librosa.feature.tonnetz(y=data, sr=sample_rate))
                    tonnetz_list.append(tonnetz)
                except Exception as e:
                    print(f"Error reading WAV file {filename}: {e}")
    except FileNotFoundError:
        print(f"Directory not found: {directory}")
    return tonnetz_list

def extract_tonnetz_features(data, sample_rate):
    tonnetz = np.array(librosa.feature.tonnetz(y=data, sr=sample_rate))
    return tonnetz

def populate_data(real_files_content, fake_files_content):
    X = []
    y = []

    for filename, content in real_files_content.items():
        tonnetz = extract_tonnetz_features(content['data'], content['sample_rate'])
        X.append(tonnetz)
        y.append(1)
    
    for filename , content in fake_files_content.items():
        tonnetz = extract_tonnetz_features(content['data'], content['sample_rate'])
        X.append(tonnetz)
        y.append(0)

    return X, y

def pad_and_stack(arrays, pad_value=0):
    max_len = max(len(arr) for arr in arrays)
    padded_arrays = [np.pad(arr, (0, max_len - len(arr)), 'constant', constant_values=pad_value) for arr in arrays]
    return np.stack(padded_arrays)

def lstm_tonnetz_model():
    real_dataset_directory = 'C:/Users/giga-/OneDrive/Desktop/Thesis/FakeOrReal dataset/for-original/training/real'
    fake_dataset_directory = 'C:/Users/giga-/OneDrive/Desktop/Thesis/FakeOrReal dataset/for-original/training/fake'

    X_real = extract_tonnetz_from_wav(real_dataset_directory)
    X_fake = extract_tonnetz_from_wav(fake_dataset_directory)

    # Find the maximum dimensions across all Tonnetz arrays
    max_rows = max(arr.shape[0] for arr in X_real + X_fake)
    max_cols = max(arr.shape[1] for arr in X_real + X_fake)

    # Pad each Tonnetz array to the maximum dimensions
    def pad_tonnetz(tonnetz, max_rows, max_cols):
        rows_pad = max_rows - tonnetz.shape[0]
        cols_pad = max_cols - tonnetz.shape[1]
        return np.pad(tonnetz, ((0, rows_pad), (0, cols_pad)), 'constant')

    X_real = [pad_tonnetz(x, max_rows, max_cols) for x in X_real]
    X_fake = [pad_tonnetz(x, max_rows, max_cols) for x in X_fake]

    X_tonnetz = X_real + X_fake
    y_tonnetz = [1] * len(X_real) + [0] * len(X_fake)

    X_tonnetz = np.array(X_tonnetz)
    y_tonnetz = np.array(y_tonnetz)

    # Reshape data to 3D array (samples, timesteps, features)
    # X_tonnetz = X_tonnetz.reshape((X_tonnetz.shape[0], X_tonnetz.shape[1], 1))

    X_train_tonnetz = X_tonnetz[:int(0.7 * len(X_tonnetz))] # 70% of the data
    y_train_tonnetz = y_tonnetz[:int(0.7 * len(y_tonnetz))]   # 70% of the data

    X_validation_tonnetz = X_tonnetz[int(0.7 * len(X_tonnetz)):int(0.85 * len(X_tonnetz))] # 15% of the data
    y_validation_tonnetz = y_tonnetz[int(0.7 * len(y_tonnetz)):int(0.85 * len(y_tonnetz))]   # 15% of the data

    X_test_tonnetz = X_tonnetz[int(0.85 * len(X_tonnetz)):] # 15% of the data
    y_test_tonnetz = y_tonnetz[int(0.85 * len(y_tonnetz)):]   # 15% of the data

    print('X_tonnetz_entry_shape:', X_tonnetz[0].shape)
    print('X_tonnetz_shape:', X_tonnetz.shape)
    # Create LSTM model with a simple architecture
    model = Sequential([
        LSTM(32),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    trained_model = model.fit(X_train_tonnetz, y_train_tonnetz, epochs=10, batch_size=32, validation_data=(X_validation_tonnetz, y_validation_tonnetz))

    # Print training results
    print("\nTraining Results:")
    print(f"Final training accuracy: {trained_model.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {trained_model.history['val_accuracy'][-1]:.4f}")
    print(f"Final training loss: {trained_model.history['loss'][-1]:.4f}")
    print(f"Final validation loss: {trained_model.history['val_loss'][-1]:.4f}")

    # Get predictions
    predictions = model.predict(X_test_tonnetz)

    print('-' * 50 + '\n')
    print('Predictions:', predictions)
    print('-' * 50 + '\n')

    accuracy_score = model.evaluate(x=X_test_tonnetz,y=y_test_tonnetz)

    print('Accuracy:', str(accuracy_score[1] * 100) + '%')

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(trained_model.history['accuracy'], label='Training Accuracy')
    plt.plot(trained_model.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(trained_model.history['loss'], label='Training Loss')
    plt.plot(trained_model.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return model

tonnetz_model_lstm = lstm_tonnetz_model()