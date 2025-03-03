import os
from scipy.io import wavfile
import numpy as np
from sklearn.preprocessing import StandardScaler
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.layers import TimeDistributed as TimeDistributed # type: ignore
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
                    tonnetz_scaled = StandardScaler().fit_transform(tonnetz)
                    tonnetz_list.append(tonnetz_scaled)
                except Exception as e:
                    print(f"Error reading WAV file {filename}: {e}")
    except FileNotFoundError:
        print(f"Directory not found: {directory}")
    return tonnetz_list

def extract_mfcc_from_wav(directory):
    mfcc_list = []
    try:
        for filename in os.listdir(directory):
            if filename.endswith('.wav'):
                filepath = os.path.join(directory, filename)
                try:
                    data, sample_rate = librosa.load(filepath)
                    mfcc = np.array(librosa.feature.mfcc(y=data, sr=sample_rate))
                    mfcc_scaled = StandardScaler().fit_transform(mfcc)
                    mfcc_list.append(mfcc_scaled)
                except Exception as e:
                    print(f"Error reading WAV file {filename}: {e}")
    except FileNotFoundError:
        print(f"Directory not found: {directory}")
    return mfcc_list

def extract_spectral_contrast_from_wav(directory):
    spectral_contrast_list = []
    try:
        for filename in os.listdir(directory):
            if filename.endswith('.wav'):
                filepath = os.path.join(directory, filename)
                try:
                    data, sample_rate = librosa.load(filepath)
                    spectral_contrast = np.array(librosa.feature.spectral_contrast(y=data, sr=sample_rate))
                    spectral_contrast_scaled = StandardScaler().fit_transform(spectral_contrast)
                    spectral_contrast_list.append(spectral_contrast_scaled)
                except Exception as e:
                    print(f"Error reading WAV file {filename}: {e}")
    except FileNotFoundError:
        print(f"Directory not found: {directory}")
    return spectral_contrast_list

def extract_spectral_centroid_from_wav(directory):
    spectral_centroid_list = []
    try:
        for filename in os.listdir(directory):
            if filename.endswith('.wav'):
                filepath = os.path.join(directory, filename)
                try:
                    data, sample_rate = librosa.load(filepath)
                    spectral_centroid = np.array(librosa.feature.spectral_centroid(y=data, sr=sample_rate))
                    spectral_centroid_scaled = StandardScaler().fit_transform(spectral_centroid)
                    spectral_centroid_list.append(spectral_centroid_scaled)
                except Exception as e:
                    print(f"Error reading WAV file {filename}: {e}")
    except FileNotFoundError:
        print(f"Directory not found: {directory}")
    return spectral_centroid_list

def extract_chroma_stft_from_wav(directory):
    chroma_stft_list = []
    try:
        for filename in os.listdir(directory):
            if filename.endswith('.wav'):
                filepath = os.path.join(directory, filename)
                try:
                    data, sample_rate = librosa.load(filepath)
                    chroma_stft = np.array(librosa.feature.chroma_stft(y=data, sr=sample_rate))
                    chroma_stft_scaled = StandardScaler().fit_transform(chroma_stft)
                    chroma_stft_list.append(chroma_stft_scaled)
                except Exception as e:
                    print(f"Error reading WAV file {filename}: {e}")
    except FileNotFoundError:
        print(f"Directory not found: {directory}")
    return chroma_stft_list

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
    X_1 = extract_mfcc_from_wav('C:/Users/giga-/OneDrive/Desktop/Thesis/FakeOrReal dataset/for-original/testing/real')
    X_2 = extract_mfcc_from_wav('C:/Users/giga-/OneDrive/Desktop/Thesis/FakeOrReal dataset/for-original/testing/fake')

    # Find the maximum dimensions across all Tonnetz arrays
    max_rows = max(arr.shape[0] for arr in X_real + X_fake + X_1 + X_2)
    max_cols = max(arr.shape[1] for arr in X_real + X_fake + X_1 + X_2)

    # Pad each Tonnetz array to the maximum dimensions
    def pad_tonnetz(tonnetz, max_rows, max_cols):
        rows_pad = max_rows - tonnetz.shape[0]
        cols_pad = max_cols - tonnetz.shape[1]
        return np.pad(tonnetz, ((0, rows_pad), (0, cols_pad)), 'constant')

    X_real = [pad_tonnetz(x, max_rows, max_cols) for x in X_real]
    X_fake = [pad_tonnetz(x, max_rows, max_cols) for x in X_fake]
    X_1 = [pad_tonnetz(x, max_rows, max_cols) for x in X_1]
    X_2 = [pad_tonnetz(x, max_rows, max_cols) for x in X_2]

    X_tonnetz = X_real + X_1 + X_fake + X_2
    y_tonnetz = [1] * (len(X_real) + len(X_1)) + [0] * (len(X_fake) + len(X_2))

    X_tonnetz = np.array(X_tonnetz)
    y_tonnetz = np.array(y_tonnetz)

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

def lstm_mfcc_model():
    real_dataset_directory = 'C:/Users/giga-/OneDrive/Desktop/Thesis/FakeOrReal dataset/for-original/training/real'
    fake_dataset_directory = 'C:/Users/giga-/OneDrive/Desktop/Thesis/FakeOrReal dataset/for-original/training/fake'

    X_real = extract_mfcc_from_wav(real_dataset_directory)
    X_fake = extract_mfcc_from_wav(fake_dataset_directory)
    X_1 = extract_mfcc_from_wav('C:/Users/giga-/OneDrive/Desktop/Thesis/FakeOrReal dataset/for-original/testing/real')
    X_2 = extract_mfcc_from_wav('C:/Users/giga-/OneDrive/Desktop/Thesis/FakeOrReal dataset/for-original/testing/fake')

    # Find the maximum dimensions across all MFCC arrays
    max_rows = max(arr.shape[0] for arr in X_real + X_fake + X_1 + X_2)
    max_cols = max(arr.shape[1] for arr in X_real + X_fake + X_1 + X_2)

    # Pad each MFCC array to the maximum dimensions
    def pad_mfcc(mfcc, max_rows, max_cols):
        rows_pad = max_rows - mfcc.shape[0]
        cols_pad = max_cols - mfcc.shape[1]
        return np.pad(mfcc, ((0, rows_pad), (0, cols_pad)), 'constant')
    
    X_real = [pad_mfcc(x, max_rows, max_cols) for x in X_real]
    X_fake = [pad_mfcc(x, max_rows, max_cols) for x in X_fake]
    X_1 = [pad_mfcc(x, max_rows, max_cols) for x in X_1]
    X_2 = [pad_mfcc(x, max_rows, max_cols) for x in X_2]

    X_mfcc = X_real + X_1 + X_fake + X_2
    y_mfcc = [1] * (len(X_real) + len(X_1)) + [0] * (len(X_fake) + len(X_2))

    X_mfcc = np.array(X_mfcc)
    y_mfcc = np.array(y_mfcc)

    X_train_mfcc = X_mfcc[:int(0.7 * len(X_mfcc))] # 70% of the data
    y_train_mfcc = y_mfcc[:int(0.7 * len(y_mfcc))]   # 70% of the data

    X_validation_mfcc = X_mfcc[int(0.7 * len(X_mfcc)):int(0.85 * len(X_mfcc))] # 15% of the data
    y_validation_mfcc = y_mfcc[int(0.7 * len(y_mfcc)):int(0.85 * len(y_mfcc))]   # 15% of the data

    X_test_mfcc = X_mfcc[int(0.85 * len(X_mfcc)):] # 15% of the data
    y_test_mfcc = y_mfcc[int(0.85 * len(y_mfcc)):]   # 15% of the data

    print('X_mfcc_entry_shape:', X_mfcc[0].shape)
    print('X_mfcc_shape:', X_mfcc.shape)

    # Create LSTM model with a simple architecture
    model = Sequential([
        LSTM(32),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    trained_model = model.fit(X_train_mfcc, y_train_mfcc, epochs=10, batch_size=32, validation_data=(X_validation_mfcc, y_validation_mfcc))

    # Print training results
    print("\nTraining Results:")
    print(f"Final training accuracy: {trained_model.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {trained_model.history['val_accuracy'][-1]:.4f}")
    print(f"Final training loss: {trained_model.history['loss'][-1]:.4f}")
    print(f"Final validation loss: {trained_model.history['val_loss'][-1]:.4f}")

    # Get predictions
    predictions = model.predict(X_test_mfcc)

    print('-' * 50 + '\n')
    print('Predictions:', predictions)
    print('-' * 50 + '\n')

    accuracy_score = model.evaluate(x=X_test_mfcc,y=y_test_mfcc)

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

def lstm_spectral_contrast_model():
    real_dataset_directory = 'C:/Users/giga-/OneDrive/Desktop/Thesis/FakeOrReal dataset/for-original/training/real'
    fake_dataset_directory = 'C:/Users/giga-/OneDrive/Desktop/Thesis/FakeOrReal dataset/for-original/training/fake'

    X_real = extract_spectral_contrast_from_wav(real_dataset_directory)
    X_fake = extract_spectral_contrast_from_wav(fake_dataset_directory)
    X_1 = extract_spectral_contrast_from_wav('C:/Users/giga-/OneDrive/Desktop/Thesis/FakeOrReal dataset/for-original/testing/real')
    X_2 = extract_spectral_contrast_from_wav('C:/Users/giga-/OneDrive/Desktop/Thesis/FakeOrReal dataset/for-original/testing/fake')

    # Find the maximum dimensions across all Spectral Contrast arrays
    max_rows = max(arr.shape[0] for arr in X_real + X_fake + X_1 + X_2)
    max_cols = max(arr.shape[1] for arr in X_real + X_fake + X_1 + X_2)

    # Pad each Spectral Contrast array to the maximum dimensions
    def pad_spectral_contrast(spectral_contrast, max_rows, max_cols):
        rows_pad = max_rows - spectral_contrast.shape[0]
        cols_pad = max_cols - spectral_contrast.shape[1]
        return np.pad(spectral_contrast, ((0, rows_pad), (0, cols_pad)), 'constant')
    
    X_real = [pad_spectral_contrast(x, max_rows, max_cols) for x in X_real]
    X_fake = [pad_spectral_contrast(x, max_rows, max_cols) for x in X_fake]
    X_1 = [pad_spectral_contrast(x, max_rows, max_cols) for x in X_1]
    X_2 = [pad_spectral_contrast(x, max_rows, max_cols) for x in X_2]

    X_spectral_contrast = X_real + X_1 + X_fake + X_2
    y_spectral_contrast = [1] * (len(X_real) + len(X_1)) + [0] * (len(X_fake) + len(X_2))

    X_spectral_contrast = np.array(X_spectral_contrast)
    y_spectral_contrast = np.array(y_spectral_contrast)

    X_train_spectral_contrast = X_spectral_contrast[:int(0.7 * len(X_spectral_contrast))] # 70% of the data
    y_train_spectral_contrast = y_spectral_contrast[:int(0.7 * len(y_spectral_contrast))]   # 70% of the data

    X_validation_spectral_contrast = X_spectral_contrast[int(0.7 * len(X_spectral_contrast)):int(0.85 * len(X_spectral_contrast))] # 15% of the data
    y_validation_spectral_contrast = y_spectral_contrast[int(0.7 * len(y_spectral_contrast)):int(0.85 * len(y_spectral_contrast))]   # 15% of the data

    X_test_spectral_contrast = X_spectral_contrast[int(0.85 * len(X_spectral_contrast)):] # 15% of the data
    y_test_spectral_contrast = y_spectral_contrast[int(0.85 * len(y_spectral_contrast)):]   # 15% of the data

    print('X_spectral_entry_shape:', X_spectral_contrast[0].shape)
    print('X_spectral_shape:', X_spectral_contrast.shape)
    
    # Create LSTM model with a simple architecture
    model = Sequential([
        LSTM(32),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    trained_model = model.fit(X_train_spectral_contrast, y_train_spectral_contrast, epochs=10, batch_size=32, validation_data=(X_validation_spectral_contrast, y_validation_spectral_contrast))

    # Print training results
    print("\nTraining Results:")
    print(f"Final training accuracy: {trained_model.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {trained_model.history['val_accuracy'][-1]:.4f}")
    print(f"Final training loss: {trained_model.history['loss'][-1]:.4f}")
    print(f"Final validation loss: {trained_model.history['val_loss'][-1]:.4f}")

    # Get predictions
    predictions = model.predict(X_test_spectral_contrast)

    print('-' * 50 + '\n')
    print('Predictions:', predictions)
    print('-' * 50 + '\n')

    accuracy_score = model.evaluate(x=X_test_spectral_contrast,y=y_test_spectral_contrast)

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

def lstm_spectral_centroid_model():
    real_dataset_directory = 'C:/Users/giga-/OneDrive/Desktop/Thesis/FakeOrReal dataset/for-original/training/real'
    fake_dataset_directory = 'C:/Users/giga-/OneDrive/Desktop/Thesis/FakeOrReal dataset/for-original/training/fake'

    X_real = extract_spectral_centroid_from_wav(real_dataset_directory)
    X_fake = extract_spectral_centroid_from_wav(fake_dataset_directory)
    X_1 = extract_spectral_centroid_from_wav('C:/Users/giga-/OneDrive/Desktop/Thesis/FakeOrReal dataset/for-original/testing/real')
    X_2 = extract_spectral_centroid_from_wav('C:/Users/giga-/OneDrive/Desktop/Thesis/FakeOrReal dataset/for-original/testing/fake')

    # Find the maximum dimensions across all Spectral Contrast arrays
    max_rows = max(arr.shape[0] for arr in X_real + X_fake + X_1 + X_2)
    max_cols = max(arr.shape[1] for arr in X_real + X_fake + X_1 + X_2)

    # Pad each Spectral Contrast array to the maximum dimensions
    def pad_spectral_centroid(spectral_centroid, max_rows, max_cols):
        rows_pad = max_rows - spectral_centroid.shape[0]
        cols_pad = max_cols - spectral_centroid.shape[1]
        return np.pad(spectral_centroid, ((0, rows_pad), (0, cols_pad)), 'constant')
    
    X_real = [pad_spectral_centroid(x, max_rows, max_cols) for x in X_real]
    X_fake = [pad_spectral_centroid(x, max_rows, max_cols) for x in X_fake]
    X_1 = [pad_spectral_centroid(x, max_rows, max_cols) for x in X_1]
    X_2 = [pad_spectral_centroid(x, max_rows, max_cols) for x in X_2]

    X_spectral_centroid = X_real + X_1 + X_fake + X_2
    y_spectral_centroid = [1] * (len(X_real) + len(X_1)) + [0] * (len(X_fake) + len(X_2))

    X_spectral_centroid = np.array(X_spectral_centroid)
    y_spectral_centroid = np.array(y_spectral_centroid)

    X_train_spectral_centroid = X_spectral_centroid[:int(0.7 * len(X_spectral_centroid))] # 70% of the data
    y_train_spectral_centroid = y_spectral_centroid[:int(0.7 * len(y_spectral_centroid))]   # 70% of the data

    X_validation_spectral_centroid = X_spectral_centroid[int(0.7 * len(X_spectral_centroid)):int(0.85 * len(X_spectral_centroid))] # 15% of the data
    y_validation_spectral_centroid = y_spectral_centroid[int(0.7 * len(y_spectral_centroid)):int(0.85 * len(y_spectral_centroid))]   # 15% of the data

    X_test_spectral_centroid = X_spectral_centroid[int(0.85 * len(X_spectral_centroid)):] # 15% of the data
    y_test_spectral_centroid = y_spectral_centroid[int(0.85 * len(y_spectral_centroid)):]   # 15% of the data

    print('X_spectral_entry_shape:', X_spectral_centroid[0].shape)
    print('X_spectral_shape:', X_spectral_centroid.shape)
    
    # Create LSTM model with a simple architecture
    model = Sequential([
        LSTM(32),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    trained_model = model.fit(X_train_spectral_centroid, y_train_spectral_centroid, epochs=10, batch_size=32, validation_data=(X_validation_spectral_centroid, y_validation_spectral_centroid))

    # Print training results
    print("\nTraining Results:")
    print(f"Final training accuracy: {trained_model.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {trained_model.history['val_accuracy'][-1]:.4f}")
    print(f"Final training loss: {trained_model.history['loss'][-1]:.4f}")
    print(f"Final validation loss: {trained_model.history['val_loss'][-1]:.4f}")

    # Get predictions
    predictions = model.predict(X_test_spectral_centroid)

    print('-' * 50 + '\n')
    print('Predictions:', predictions)
    print('-' * 50 + '\n')

    accuracy_score = model.evaluate(x=X_test_spectral_centroid,y=y_test_spectral_centroid)

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

def lstm_chroma_stft_model():

    real_dataset_directory = 'C:/Users/giga-/OneDrive/Desktop/Thesis/FakeOrReal dataset/for-original/training/real'
    fake_dataset_directory = 'C:/Users/giga-/OneDrive/Desktop/Thesis/FakeOrReal dataset/for-original/training/fake'

    X_real = extract_chroma_stft_from_wav(real_dataset_directory)
    X_fake = extract_chroma_stft_from_wav(fake_dataset_directory)
    X_1 = extract_chroma_stft_from_wav('C:/Users/giga-/OneDrive/Desktop/Thesis/FakeOrReal dataset/for-original/testing/real')
    X_2 = extract_chroma_stft_from_wav('C:/Users/giga-/OneDrive/Desktop/Thesis/FakeOrReal dataset/for-original/testing/fake')

    # Find the maximum dimensions across all Chroma STFT arrays
    max_rows = max(arr.shape[0] for arr in X_real + X_fake + X_1 + X_2)
    max_cols = max(arr.shape[1] for arr in X_real + X_fake + X_1 + X_2)

    # Pad each Chroma STFT array to the maximum dimensions
    def pad_chroma_stft(chroma_stft, max_rows, max_cols):
        rows_pad = max_rows - chroma_stft.shape[0]
        cols_pad = max_cols - chroma_stft.shape[1]
        return np.pad(chroma_stft, ((0, rows_pad), (0, cols_pad)), 'constant')
    
    X_real = [pad_chroma_stft(x, max_rows, max_cols) for x in X_real]
    X_fake = [pad_chroma_stft(x, max_rows, max_cols) for x in X_fake]
    X_1 = [pad_chroma_stft(x, max_rows, max_cols) for x in X_1]
    X_2 = [pad_chroma_stft(x, max_rows, max_cols) for x in X_2]

    X_chroma_stft = X_real + X_1 + X_fake + X_2
    y_chroma_stft = [1] * (len(X_real) + len(X_1)) + [0] * (len(X_fake) + len(X_2))

    X_chroma_stft = np.array(X_chroma_stft)
    y_chroma_stft = np.array(y_chroma_stft)

    X_train_chroma_stft = X_chroma_stft[:int(0.7 * len(X_chroma_stft))] # 70% of the data
    y_train_chroma_stft = y_chroma_stft[:int(0.7 * len(y_chroma_stft))]   # 70% of the data

    X_validation_chroma_stft = X_chroma_stft[int(0.7 * len(X_chroma_stft)):int(0.85 * len(X_chroma_stft))] # 15% of the data
    y_validation_chroma_stft = y_chroma_stft[int(0.7 * len(y_chroma_stft)):int(0.85 * len(y_chroma_stft))]   # 15% of the data

    X_test_chroma_stft = X_chroma_stft[int(0.85 * len(X_chroma_stft)):] # 15% of the data
    y_test_chroma_stft = y_chroma_stft[int(0.85 * len(y_chroma_stft)):]   # 15% of the data

    print('X_chroma_entry_shape:', X_chroma_stft[0].shape)
    print('X_chroma_shape:', X_chroma_stft.shape)

    # Create LSTM model with a simple architecture
    model = Sequential([
        LSTM(32),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    trained_model = model.fit(X_train_chroma_stft, y_train_chroma_stft, epochs=10, batch_size=32, validation_data=(X_validation_chroma_stft, y_validation_chroma_stft))

    # Print training results
    print("\nTraining Results:")
    print(f"Final training accuracy: {trained_model.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {trained_model.history['val_accuracy'][-1]:.4f}")
    print(f"Final training loss: {trained_model.history['loss'][-1]:.4f}")
    print(f"Final validation loss: {trained_model.history['val_loss'][-1]:.4f}")

    # Get predictions
    predictions = model.predict(X_test_chroma_stft)

    print('-' * 50 + '\n')
    print('Predictions:', predictions)
    print('-' * 50 + '\n')

    accuracy_score = model.evaluate(x=X_test_chroma_stft,y=y_test_chroma_stft)

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

def cnn_tonnetz_model():
    real_dataset_directory = 'C:/Users/giga-/OneDrive/Desktop/Thesis/FakeOrReal dataset/for-original/training/real'
    fake_dataset_directory = 'C:/Users/giga-/OneDrive/Desktop/Thesis/FakeOrReal dataset/for-original/training/fake'

    X_real = extract_tonnetz_from_wav(real_dataset_directory)
    X_fake = extract_tonnetz_from_wav(fake_dataset_directory)
    X_1 = extract_tonnetz_from_wav('C:/Users/giga-/OneDrive/Desktop/Thesis/FakeOrReal dataset/for-original/testing/real')
    X_2 = extract_tonnetz_from_wav('C:/Users/giga-/OneDrive/Desktop/Thesis/FakeOrReal dataset/for-original/testing/fake')

    # Find the maximum dimensions across all Tonnetz arrays
    max_rows = max(arr.shape[0] for arr in X_real + X_fake + X_1 + X_2)
    max_cols = max(arr.shape[1] for arr in X_real + X_fake + X_1 + X_2)

    # Pad each Tonnetz array to the maximum dimensions
    def pad_tonnetz(tonnetz, max_rows, max_cols):
        rows_pad = max_rows - tonnetz.shape[0]
        cols_pad = max_cols - tonnetz.shape[1]
        return np.pad(tonnetz, ((0, rows_pad), (0, cols_pad)), 'constant')

    X_real = [pad_tonnetz(x, max_rows, max_cols) for x in X_real]
    X_fake = [pad_tonnetz(x, max_rows, max_cols) for x in X_fake]
    X_1 = [pad_tonnetz(x, max_rows, max_cols) for x in X_1]
    X_2 = [pad_tonnetz(x, max_rows, max_cols) for x in X_2]

    X_tonnetz = X_real + X_1 + X_fake + X_2
    y_tonnetz = [1] * (len(X_real) + len(X_1)) + [0] * (len(X_fake) + len(X_2))

    X_tonnetz = np.array(X_tonnetz)
    y_tonnetz = np.array(y_tonnetz)

    X_tonnetz = X_tonnetz[..., np.newaxis]
    X_tonnetz = np.expand_dims(X_tonnetz, axis=1)

    X_train_tonnetz = X_tonnetz[:int(0.7 * len(X_tonnetz))] # 70% of the data
    y_train_tonnetz = y_tonnetz[:int(0.7 * len(y_tonnetz))]   # 70% of the data

    X_validation_tonnetz = X_tonnetz[int(0.7 * len(X_tonnetz)):int(0.85 * len(X_tonnetz))] # 15% of the data
    y_validation_tonnetz = y_tonnetz[int(0.7 * len(y_tonnetz)):int(0.85 * len(y_tonnetz))]   # 15% of the data

    X_test_tonnetz = X_tonnetz[int(0.85 * len(X_tonnetz)):] # 15% of the data
    y_test_tonnetz = y_tonnetz[int(0.85 * len(y_tonnetz)):]   # 15% of the data

    print('X_tonnetz_entry_shape:', X_tonnetz[0].shape)
    print('X_tonnetz_shape:', X_tonnetz.shape)

    # Define the CNN model
    model = Sequential([
        TimeDistributed(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(None, max_rows, max_cols, 1), padding='same')),
        TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2), padding='same')),
        TimeDistributed(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')),
        TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2))),
        TimeDistributed(tf.keras.layers.Flatten()),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    trained_model = model.fit(X_train_tonnetz, y_train_tonnetz, epochs=10, batch_size=32, validation_data=(X_validation_tonnetz, y_validation_tonnetz))

    # Print training results
    print("\nTraining Results:")
    print(f"Final training accuracy: {trained_model.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {trained_model.history['val_accuracy'][-1]:.4f}")
    print(f"Final training loss: {trained_model.history['loss'][-1]:.4f}")
    print(f"Final validation loss: {trained_model.history['val_loss'][-1]:.4f}")

    # Evaluate the model
    accuracy_score = model.evaluate(X_test_tonnetz, y_test_tonnetz)
    print('Accuracy:', str(accuracy_score[1] * 100) + '%')

    # Plot training history
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

def cnn_mfcc_model():
    real_dataset_directory = 'C:/Users/giga-/OneDrive/Desktop/Thesis/FakeOrReal dataset/for-original/training/real'
    fake_dataset_directory = 'C:/Users/giga-/OneDrive/Desktop/Thesis/FakeOrReal dataset/for-original/training/fake'

    X_real = extract_mfcc_from_wav(real_dataset_directory)
    X_fake = extract_mfcc_from_wav(fake_dataset_directory)
    X_1 = extract_mfcc_from_wav('C:/Users/giga-/OneDrive/Desktop/Thesis/FakeOrReal dataset/for-original/testing/real')
    X_2 = extract_mfcc_from_wav('C:/Users/giga-/OneDrive/Desktop/Thesis/FakeOrReal dataset/for-original/testing/fake')

    # Find the maximum dimensions across all Tonnetz arrays
    max_rows = max(arr.shape[0] for arr in X_real + X_fake + X_1 + X_2)
    max_cols = max(arr.shape[1] for arr in X_real + X_fake + X_1 + X_2)

    # Pad each Tonnetz array to the maximum dimensions
    def pad_mfcc(mfcc, max_rows, max_cols):
        rows_pad = max_rows - mfcc.shape[0]
        cols_pad = max_cols - mfcc.shape[1]
        return np.pad(mfcc, ((0, rows_pad), (0, cols_pad)), 'constant')

    X_real = [pad_mfcc(x, max_rows, max_cols) for x in X_real]
    X_fake = [pad_mfcc(x, max_rows, max_cols) for x in X_fake]
    X_1 = [pad_mfcc(x, max_rows, max_cols) for x in X_1]
    X_2 = [pad_mfcc(x, max_rows, max_cols) for x in X_2]

    X_mfcc = X_real + X_1 + X_fake + X_2
    y_mfcc = [1] * (len(X_real) + len(X_1)) + [0] * (len(X_fake) + len(X_2))

    X_mfcc = np.array(X_mfcc)
    y_mfcc = np.array(y_mfcc)

    X_mfcc = X_mfcc[..., np.newaxis]
    X_mfcc = np.expand_dims(X_mfcc, axis=1)

    X_train_mfcc = X_mfcc[:int(0.7 * len(X_mfcc))] # 70% of the data
    y_train_mfcc = y_mfcc[:int(0.7 * len(y_mfcc))]   # 70% of the data

    X_validation_mfcc = X_mfcc[int(0.7 * len(X_mfcc)):int(0.85 * len(X_mfcc))] # 15% of the data
    y_validation_mfcc = y_mfcc[int(0.7 * len(y_mfcc)):int(0.85 * len(y_mfcc))]   # 15% of the data

    X_test_mfcc = X_mfcc[int(0.85 * len(X_mfcc)):] # 15% of the data
    y_test_mfcc = y_mfcc[int(0.85 * len(y_mfcc)):]   # 15% of the data

    print('X_tonnetz_entry_shape:', X_mfcc[0].shape)
    print('X_tonnetz_shape:', X_mfcc.shape)

    # Define the CNN model
    model = Sequential([
        TimeDistributed(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(None, max_rows, max_cols, 1), padding='same')),
        TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2), padding='same')),
        TimeDistributed(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')),
        TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2))),
        TimeDistributed(tf.keras.layers.Flatten()),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    trained_model = model.fit(X_train_mfcc, y_train_mfcc, epochs=10, batch_size=32, validation_data=(X_validation_mfcc, y_validation_mfcc))

    # Print training results
    print("\nTraining Results:")
    print(f"Final training accuracy: {trained_model.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {trained_model.history['val_accuracy'][-1]:.4f}")
    print(f"Final training loss: {trained_model.history['loss'][-1]:.4f}")
    print(f"Final validation loss: {trained_model.history['val_loss'][-1]:.4f}")

    # Evaluate the model
    accuracy_score = model.evaluate(X_test_mfcc, y_test_mfcc)
    print('Accuracy:', str(accuracy_score[1] * 100) + '%')

    # Plot training history
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

def cnn_spectral_contrast_model():
    real_dataset_directory = 'C:/Users/giga-/OneDrive/Desktop/Thesis/FakeOrReal dataset/for-original/training/real'
    fake_dataset_directory = 'C:/Users/giga-/OneDrive/Desktop/Thesis/FakeOrReal dataset/for-original/training/fake'

    X_real = extract_spectral_contrast_from_wav(real_dataset_directory)
    X_fake = extract_spectral_contrast_from_wav(fake_dataset_directory)
    X_1 = extract_spectral_contrast_from_wav('C:/Users/giga-/OneDrive/Desktop/Thesis/FakeOrReal dataset/for-original/testing/real')
    X_2 = extract_spectral_contrast_from_wav('C:/Users/giga-/OneDrive/Desktop/Thesis/FakeOrReal dataset/for-original/testing/fake')

    # Find the maximum dimensions across all Tonnetz arrays
    max_rows = max(arr.shape[0] for arr in X_real + X_fake + X_1 + X_2)
    max_cols = max(arr.shape[1] for arr in X_real + X_fake + X_1 + X_2)

    # Pad each Tonnetz array to the maximum dimensions
    def pad_spectral_contrast(spectral_contrast, max_rows, max_cols):
        rows_pad = max_rows - spectral_contrast.shape[0]
        cols_pad = max_cols - spectral_contrast.shape[1]
        return np.pad(spectral_contrast, ((0, rows_pad), (0, cols_pad)), 'constant')

    X_real = [pad_spectral_contrast(x, max_rows, max_cols) for x in X_real]
    X_fake = [pad_spectral_contrast(x, max_rows, max_cols) for x in X_fake]
    X_1 = [pad_spectral_contrast(x, max_rows, max_cols) for x in X_1]
    X_2 = [pad_spectral_contrast(x, max_rows, max_cols) for x in X_2]

    X_spectral_contrast = X_real + X_1 + X_fake + X_2
    y_spectral_contrast = [1] * (len(X_real) + len(X_1)) + [0] * (len(X_fake) + len(X_2))

    X_spectral_contrast = np.array(X_spectral_contrast)
    y_spectral_contrast = np.array(y_spectral_contrast)

    X_spectral_contrast = X_spectral_contrast[..., np.newaxis]
    X_spectral_contrast = np.expand_dims(X_spectral_contrast, axis=1)

    X_train_spectral_contrast = X_spectral_contrast[:int(0.7 * len(X_spectral_contrast))] # 70% of the data
    y_train_spectral_contrast = y_spectral_contrast[:int(0.7 * len(y_spectral_contrast))]   # 70% of the data

    X_validation_spectral_contrast = X_spectral_contrast[int(0.7 * len(X_spectral_contrast)):int(0.85 * len(X_spectral_contrast))] # 15% of the data
    y_validation_spectral_contrast = y_spectral_contrast[int(0.7 * len(y_spectral_contrast)):int(0.85 * len(y_spectral_contrast))]   # 15% of the data

    X_test_spectral_contrast = X_spectral_contrast[int(0.85 * len(X_spectral_contrast)):] # 15% of the data
    y_test_spectral_contrast = y_spectral_contrast[int(0.85 * len(y_spectral_contrast)):]   # 15% of the data

    print('X_tonnetz_entry_shape:', X_spectral_contrast[0].shape)
    print('X_tonnetz_shape:', X_spectral_contrast.shape)

    # Define the CNN model
    model = Sequential([
        TimeDistributed(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(None, max_rows, max_cols, 1), padding='same')),
        TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2), padding='same')),
        TimeDistributed(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')),
        TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2))),
        TimeDistributed(tf.keras.layers.Flatten()),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    trained_model = model.fit(X_train_spectral_contrast, y_train_spectral_contrast, epochs=10, batch_size=32, validation_data=(X_validation_spectral_contrast, y_validation_spectral_contrast))

    # Print training results
    print("\nTraining Results:")
    print(f"Final training accuracy: {trained_model.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {trained_model.history['val_accuracy'][-1]:.4f}")
    print(f"Final training loss: {trained_model.history['loss'][-1]:.4f}")
    print(f"Final validation loss: {trained_model.history['val_loss'][-1]:.4f}")

    # Evaluate the model
    accuracy_score = model.evaluate(X_test_spectral_contrast, y_test_spectral_contrast)
    print('Accuracy:', str(accuracy_score[1] * 100) + '%')

    # Plot training history
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

def cnn_spectral_centroid_model():
    # spectral centroid doesnt seem to be a good feature
    real_dataset_directory = 'C:/Users/giga-/OneDrive/Desktop/Thesis/FakeOrReal dataset/for-original/training/real'
    fake_dataset_directory = 'C:/Users/giga-/OneDrive/Desktop/Thesis/FakeOrReal dataset/for-original/training/fake'

    X_real = extract_spectral_centroid_from_wav(real_dataset_directory)
    X_fake = extract_spectral_centroid_from_wav(fake_dataset_directory)
    X_1 = extract_spectral_centroid_from_wav('C:/Users/giga-/OneDrive/Desktop/Thesis/FakeOrReal dataset/for-original/testing/real')
    X_2 = extract_spectral_centroid_from_wav('C:/Users/giga-/OneDrive/Desktop/Thesis/FakeOrReal dataset/for-original/testing/fake')

    # Find the maximum dimensions across all Tonnetz arrays
    max_rows = max(arr.shape[0] for arr in X_real + X_fake + X_1 + X_2)
    max_cols = max(arr.shape[1] for arr in X_real + X_fake + X_1 + X_2)

    # Pad each Tonnetz array to the maximum dimensions
    def pad_spectral_centroid(spectral_centroid, max_rows, max_cols):
        rows_pad = max_rows - spectral_centroid.shape[0]
        cols_pad = max_cols - spectral_centroid.shape[1]
        return np.pad(spectral_centroid, ((0, rows_pad), (0, cols_pad)), 'constant')

    X_real = [pad_spectral_centroid(x, max_rows, max_cols) for x in X_real]
    X_fake = [pad_spectral_centroid(x, max_rows, max_cols) for x in X_fake]
    X_1 = [pad_spectral_centroid(x, max_rows, max_cols) for x in X_1]
    X_2 = [pad_spectral_centroid(x, max_rows, max_cols) for x in X_2]

    X_spectral_centroid = X_real + X_1 + X_fake  + X_2
    y_spectral_centroid = [1] * (len(X_real) + len(X_1)) + [0] * (len(X_fake) + len(X_2))

    X_spectral_centroid = np.array(X_spectral_centroid)
    y_spectral_centroid = np.array(y_spectral_centroid)

    X_spectral_centroid = X_spectral_centroid[..., np.newaxis]

    X_train_spectral_centroid = X_spectral_centroid[:int(0.7 * len(X_spectral_centroid))] # 70% of the data
    y_train_spectral_centroid = y_spectral_centroid[:int(0.7 * len(y_spectral_centroid))]   # 70% of the data

    X_validation_spectral_centroid = X_spectral_centroid[int(0.7 * len(X_spectral_centroid)):int(0.85 * len(X_spectral_centroid))] # 15% of the data
    y_validation_spectral_centroid = y_spectral_centroid[int(0.7 * len(y_spectral_centroid)):int(0.85 * len(y_spectral_centroid))]   # 15% of the data

    X_test_spectral_centroid = X_spectral_centroid[int(0.85 * len(X_spectral_centroid)):] # 15% of the data
    y_test_spectral_centroid = y_spectral_centroid[int(0.85 * len(y_spectral_centroid)):]   # 15% of the data

    print('X_tonnetz_entry_shape:', X_spectral_centroid[0].shape)
    print('X_tonnetz_shape:', X_spectral_centroid.shape)

    # Define the CNN model
    model = Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(max_rows, max_cols, 1), padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    trained_model = model.fit(X_train_spectral_centroid, y_train_spectral_centroid, epochs=10, batch_size=32, validation_data=(X_validation_spectral_centroid, y_validation_spectral_centroid))

    # Print training results
    print("\nTraining Results:")
    print(f"Final training accuracy: {trained_model.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {trained_model.history['val_accuracy'][-1]:.4f}")
    print(f"Final training loss: {trained_model.history['loss'][-1]:.4f}")
    print(f"Final validation loss: {trained_model.history['val_loss'][-1]:.4f}")

    # Evaluate the model
    accuracy_score = model.evaluate(X_test_spectral_centroid, y_test_spectral_centroid)
    print('Accuracy:', str(accuracy_score[1] * 100) + '%')

    # Plot training history
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

def cnn_chroma_stft_model():
    real_dataset_directory = 'C:/Users/giga-/OneDrive/Desktop/Thesis/FakeOrReal dataset/for-original/training/real'
    fake_dataset_directory = 'C:/Users/giga-/OneDrive/Desktop/Thesis/FakeOrReal dataset/for-original/training/fake'

    X_real = extract_chroma_stft_from_wav(real_dataset_directory)
    X_fake = extract_chroma_stft_from_wav(fake_dataset_directory)
    X_1 = extract_chroma_stft_from_wav('C:/Users/giga-/OneDrive/Desktop/Thesis/FakeOrReal dataset/for-original/testing/real')
    X_2 = extract_chroma_stft_from_wav('C:/Users/giga-/OneDrive/Desktop/Thesis/FakeOrReal dataset/for-original/testing/fake')

    # Find the maximum dimensions across all Chroma STFT arrays
    max_rows = max(arr.shape[0] for arr in X_real + X_fake + X_1 + X_2)
    max_cols = max(arr.shape[1] for arr in X_real + X_fake + X_1 + X_2)

    # Pad each Chroma STFT array
    def pad_chroma_stft(chroma_stft, max_rows, max_cols):
        rows_pad = max_rows - chroma_stft.shape[0]
        cols_pad = max_cols - chroma_stft.shape[1]
        return np.pad(chroma_stft, ((0, rows_pad), (0, cols_pad)), 'constant')

    X_real = [pad_chroma_stft(x, max_rows, max_cols) for x in X_real]
    X_fake = [pad_chroma_stft(x, max_rows, max_cols) for x in X_fake]
    X_1 = [pad_chroma_stft(x, max_rows, max_cols) for x in X_1]
    X_2 = [pad_chroma_stft(x, max_rows, max_cols) for x in X_2]

    X_chroma_stft = X_real + X_1 + X_fake + X_2
    y_chroma_stft = [1] * (len(X_real) + len(X_1)) + [0] * (len(X_fake) + len(X_2))

    X_chroma_stft = np.array(X_chroma_stft)
    y_chroma_stft = np.array(y_chroma_stft)

    X_chroma_stft = X_chroma_stft[..., np.newaxis]
    X_chroma_stft = np.expand_dims(X_chroma_stft, axis=1)

    X_train_chroma_stft = X_chroma_stft[:int(0.7 * len(X_chroma_stft))] # 70% of the data
    y_train_chroma_stft = y_chroma_stft[:int(0.7 * len(y_chroma_stft))]

    X_validation_chroma_stft = X_chroma_stft[int(0.7 * len(X_chroma_stft)):int(0.85 * len(X_chroma_stft))] # 15% of the data
    y_validation_chroma_stft = y_chroma_stft[int(0.7 * len(y_chroma_stft)):int(0.85 * len(y_chroma_stft))]

    X_test_chroma_stft = X_chroma_stft[int(0.85 * len(X_chroma_stft)):] # 15% of the data
    y_test_chroma_stft = y_chroma_stft[int(0.85 * len(y_chroma_stft)):]

    print('X_chroma_entry_shape:', X_chroma_stft[0].shape)
    print('X_chroma_shape:', X_chroma_stft.shape)

    # Define the CNN model
    model = Sequential([
        TimeDistributed(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(None, max_rows, max_cols, 1), padding='same')),
        TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2), padding='same')),
        TimeDistributed(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')),
        TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2))),
        TimeDistributed(tf.keras.layers.Flatten()),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    trained_model = model.fit(X_train_chroma_stft, y_train_chroma_stft, epochs=10, batch_size=32, validation_data=(X_validation_chroma_stft, y_validation_chroma_stft))

    # Print training results
    print("\nTraining Results:")
    print(f"Final training accuracy: {trained_model.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {trained_model.history['val_accuracy'][-1]:.4f}")
    print(f"Final training loss: {trained_model.history['loss'][-1]:.4f}")
    print(f"Final validation loss: {trained_model.history['val_loss'][-1]:.4f}")

    # Evaluate the model
    accuracy_score = model.evaluate(X_test_chroma_stft, y_test_chroma_stft)
    
    print('Accuracy:', str(accuracy_score[1] * 100) + '%')

    # Plot training history
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

# tonnetz_model_lstm = lstm_tonnetz_model()
# mfcc_model_lstm = lstm_mfcc_model()
# spectral_contrast_model_lstm = lstm_spectral_contrast_model()
# spectral_centroid_model_lstm = lstm_spectral_centroid_model() # 0 accuracy with full files
# chroma_stft_model_lstm = lstm_chroma_stft_model()
# cnn_model_tonnetz = cnn_tonnetz_model() # ~45% accuracy
# cnn_model_mfcc = cnn_mfcc_model()
# cnn_model_spectral_contrast = cnn_spectral_contrast_model()
# cnn_model_spectral_centroid = cnn_spectral_centroid_model() # 0 accuracy with full files
# cnn_model_chroma_stft = cnn_chroma_stft_model()