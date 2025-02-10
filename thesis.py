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
                    tonnetz = librosa.feature.tonnetz(y=data, sr=sample_rate)
                    # Transpose to get time steps as the first dimension
                    tonnetz = tonnetz.T
                    tonnetz_list.append(tonnetz)
                except Exception as e:
                    print(f"Error reading WAV file {filename}: {e}")
    except FileNotFoundError:
        print(f"Directory not found: {directory}")
    return tonnetz_list

def extract_tonnetz_features(data, sample_rate):
    tonnetz = librosa.feature.tonnetz(y=data, sr=sample_rate)
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

    X_train_tonnetz = extract_tonnetz_from_wav(real_dataset_directory)
    y_train_tonnetz = [1] * len(X_train_tonnetz)
    X_train_tonnetz.extend(extract_tonnetz_from_wav(fake_dataset_directory))
    y_train_tonnetz.extend([0] * (len(X_train_tonnetz) - len(y_train_tonnetz)))

    # LSTM (Long Short-Term Memory) layers are designed to work with sequential data, and their input shape follows this pattern: (batch_size, timesteps, features)

    # transform the data so that all arrays have the same length
    # X_train_tonnetz = pad_and_stack(X_train_tonnetz)

    # Convert training data to numpy arrays
    # X_train_tonnetz = np.array(X_train_tonnetz, dtype=np.float32)
    # y_train_tonnetz = np.array(y_train_tonnetz, dtype=np.int32)

    # Create LSTM model with a simple architecture
    model = Sequential([
        LSTM(32),  # 6 is the number of features in tonnetz
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Load and process validation data from different directories
    val_real_dir = 'C:/Users/giga-/OneDrive/Desktop/Thesis/FakeOrReal dataset/for-original/validation/real'
    val_fake_dir = 'C:/Users/giga-/OneDrive/Desktop/Thesis/FakeOrReal dataset/for-original/validation/fake'

    X_validation_tonnetz = extract_tonnetz_from_wav(val_real_dir)
    y_validation_tonnetz = [1] * len(X_validation_tonnetz)
    X_validation_tonnetz.extend(extract_tonnetz_from_wav(val_fake_dir))
    y_validation_tonnetz.extend([0] * (len(X_validation_tonnetz) - len(y_validation_tonnetz)))

    # transform the data so that all arrays have the same length
    # X_validation_tonnetz = pad_and_stack(X_validation_tonnetz)

    # print(X_train_tonnetz.shape)
    # print('-' * 50 + '\n')
    # print(X_validation_tonnetz.shape)

    trained_model = model.fit(X_train_tonnetz, y_train_tonnetz, epochs=10, batch_size=32, validation_data=(X_validation_tonnetz, y_validation_tonnetz))

    # Print training results
    print("\nTraining Results:")
    print(f"Final training accuracy: {trained_model.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {trained_model.history['val_accuracy'][-1]:.4f}")
    print(f"Final training loss: {trained_model.history['loss'][-1]:.4f}")
    print(f"Final validation loss: {trained_model.history['val_loss'][-1]:.4f}")

    # Load and process testing data from different directories
    real_test_dir = 'C:/Users/giga-/OneDrive/Desktop/Thesis/FakeOrReal dataset/for-original/testing/real'
    fake_test_dir = 'C:/Users/giga-/OneDrive/Desktop/Thesis/FakeOrReal dataset/for-original/testing/fake'

    X_test_tonnetz = extract_tonnetz_from_wav(real_test_dir)
    X_test_tonnetz.extend(extract_tonnetz_from_wav(fake_test_dir))

    # transform the data so that all arrays have the same length
    # X_test_tonnetz = pad_and_stack(X_test_tonnetz)

    # Convert to numpy array and ensure correct type
    # X_test_tonnetz = np.array(X_test_tonnetz, dtype=np.float32)

    # Get predictions
    predictions = model.predict(X_test_tonnetz)

    print('-' * 50 + '\n')
    print('Predictions:', predictions)
    print('-' * 50 + '\n')

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