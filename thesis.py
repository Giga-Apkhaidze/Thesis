import os
from scipy.io import wavfile
import numpy as np
from sklearn.preprocessing import StandardScaler
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.layers import TimeDistributed as TimeDistributed # type: ignore
import matplotlib.pyplot as plt
import time

start_time = time.time()

audio_files_directory = 'C:/Users/giga-/OneDrive/Desktop/Thesis/FakeOrReal dataset/'

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
                    duration = librosa.get_duration(y=data, sr=sample_rate)
                    for start in range(0, int(duration), 1):
                        end = start + 1
                        if end > duration:
                            break
                        interval_data = data[int(start * sample_rate):int(end * sample_rate)]
                        tonnetz = np.array(librosa.feature.tonnetz(y=interval_data, sr=sample_rate))
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

def extract_mfcc_features(data, sample_rate):
    mfcc = np.array(librosa.feature.mfcc(y=data, sr=sample_rate))
    return mfcc

def extract_spectral_contrast_features(data, sample_rate):
    spectral_contrast = np.array(librosa.feature.spectral_contrast(y=data, sr=sample_rate))
    return spectral_contrast

def extract_spectral_centroid_features(data, sample_rate):
    spectral_centroid = np.array(librosa.feature.spectral_centroid(y=data, sr=sample_rate))
    return spectral_centroid

def extract_chroma_stft_features(data, sample_rate):
    chroma_stft = np.array(librosa.feature.chroma_stft(y=data, sr=sample_rate))
    return chroma_stft

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

def cnn_lstm_model(X, y, data_size, train_size):
    # Reshape the data for the CNN-LSTM model
    X = X[..., np.newaxis]
    X = np.expand_dims(X, axis=1)

    # Split the data into train and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=train_size, random_state=42)
    X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

    # Define the CNN model
    model = Sequential([
        TimeDistributed(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(None, data_size[0], data_size[1], 1), padding='same')),
        TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2), padding='same')),
        TimeDistributed(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')),
        TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2), padding='same')),
        TimeDistributed(tf.keras.layers.Flatten()),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate the model
    accuracy_score = model.evaluate(x=X_validation, y=y_validation)

    return model, accuracy_score[1]

def lstm_model(X, y, train_size):

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=train_size, random_state=42)
    X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)
    
    # Create LSTM model with a simple architecture
    model = Sequential([
        LSTM(32),
        Dense(1, activation='sigmoid'),
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    accuracy_score = model.evaluate(x=X_validation, y=y_validation)

    return model, accuracy_score[1]

def pad_array(array, max_rows, max_cols):
    rows_pad = max_rows - array.shape[0]
    cols_pad = max_cols - array.shape[1]
    return np.pad(array, ((0, rows_pad), (0, cols_pad)), 'constant')

def generate_data_CNN(directory, feature_extractors):
    X_all = {key: [] for key in feature_extractors.keys()}
    y = []
    test_files_all = {key: [] for key in feature_extractors.keys()}
    test_file_labels = []
    max_rows = 0
    max_cols = 0
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav') and 'validation' not in subdir:
                filepath = os.path.join(subdir, file)
                label = 1 if 'real' in subdir else 0
                try:
                    data, sample_rate = librosa.load(filepath)
                    for key, feature_extractor in feature_extractors.items():
                        feature = feature_extractor(data, sample_rate)
                        feature_scaled = StandardScaler().fit_transform(feature)
                        max_rows = max(max_rows, feature_scaled.shape[0])
                        max_cols = max(max_cols, feature_scaled.shape[1])
                        X_all[key].append(feature_scaled)
                    y.append(label)
                except Exception as e:
                    print(f"Error processing file {filepath}: {e}")
            elif file.endswith('.wav') and 'validation' in subdir:
                filepath = os.path.join(subdir, file)
                label = 1 if 'real' in subdir else 0
                try:
                    data, sample_rate = librosa.load(filepath)
                    for key, feature_extractor in feature_extractors.items():
                        feature = feature_extractor(data, sample_rate)
                        test_files_all[key].append(feature)
                    test_file_labels.append(label)
                except Exception as e:
                    print(f"Error processing file {filepath}: {e}")
    
    for key in X_all.keys():
        X_all[key] = [pad_array(x, max_rows, max_cols) for x in X_all[key]]
        X_all[key] = np.array(X_all[key])

        test_files_all[key] = [pad_array(file, max_rows, max_cols) for file in test_files_all[key]]
        test_files_all[key] = np.array(test_files_all[key])
        test_files_all[key] = test_files_all[key][..., np.newaxis]
        test_files_all[key] = np.expand_dims(test_files_all[key], axis=1)

    y = np.array(y)
    test_file_labels = np.array(test_file_labels)

    return X_all, y, (max_rows, max_cols), test_files_all, test_file_labels

def generate_data_LSTM(directory, feature_extractors):
    X_all = {key: [] for key in feature_extractors.keys()}
    y = []
    test_files_all = {key: [] for key in feature_extractors.keys()}
    test_files_labels = []
    rows = {key: 0 for key in feature_extractors.keys()}
    columns = {key: 0 for key in feature_extractors.keys()}
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav') and 'validation' not in subdir:
                filepath = os.path.join(subdir, file)
                label = 1 if 'real' in subdir else 0
                try:
                    data, sample_rate = librosa.load(filepath)
                    duration = librosa.get_duration(y=data, sr=sample_rate)
                    end = 0
                    for start in range(0, int(duration), 1):
                        end = start + 1
                        if end > duration:
                            break
                        interval_data = data[int(start * sample_rate):int(end * sample_rate)]
                        for key, feature_extractor in feature_extractors.items():
                            feature = feature_extractor(interval_data, sample_rate)
                            feature_scaled = StandardScaler().fit_transform(feature)
                            rows[key] = max(rows[key], feature_scaled.shape[0])
                            columns[key] = max(columns[key], feature_scaled.shape[1])
                            X_all[key].append(feature_scaled)
                        y.append(label)

                    interval_data = data[int(end * sample_rate):]
                    for key, feature_extractor in feature_extractors.items():
                        feature = feature_extractor(interval_data, sample_rate)
                        feature_scaled = StandardScaler().fit_transform(feature)
                        feature_scaled = pad_array(feature_scaled, rows[key], columns[key])
                        X_all[key].append(feature_scaled)
                    y.append(label)
                except Exception as e:
                    print(f"Error processing file {filepath}: {e}")
            elif file.endswith('.wav') and 'validation' in subdir:
                filepath = os.path.join(subdir, file)
                label = 1 if 'real' in subdir else 0
                try:
                    data, sample_rate = librosa.load(filepath)
                    duration = librosa.get_duration(y=data, sr=sample_rate)
                    end = 0
                    for start in range(0, int(duration), 1):
                        end = start + 1
                        if end > duration:
                            break
                        interval_data = data[int(start * sample_rate):int(end * sample_rate)]
                        for key, feature_extractor in feature_extractors.items():
                            feature = feature_extractor(interval_data, sample_rate)
                            feature_scaled = StandardScaler().fit_transform(feature)
                            rows[key] = max(rows[key], feature_scaled.shape[0])
                            columns[key] = max(columns[key], feature_scaled.shape[1])
                            test_files_all[key].append((feature_scaled, file))
                        test_files_labels.append(label)

                    interval_data = data[int(end * sample_rate):]
                    for key, feature_extractor in feature_extractors.items():
                        feature = feature_extractor(interval_data, sample_rate)
                        feature_scaled = StandardScaler().fit_transform(feature)
                        feature_scaled = pad_array(feature_scaled, rows[key], columns[key])
                        test_files_all[key].append((feature_scaled, file))
                    test_files_labels.append(label)
                except Exception as e:
                    print(f"Error processing file {filepath}: {e}")

    for key in X_all.keys():
        X_all[key] = np.array(X_all[key])
        test_files_all[key] = ((np.array([file[0] for file in test_files_all[key]]), [file[1] for file in test_files_all[key]]))

    return X_all, np.array(y), test_files_all, np.array(test_files_labels)

def extract_feature(directory, feature_extractor, max_rows, max_cols):
    feature_array = []
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav') and 'validation' in subdir:
                filepath = os.path.join(subdir, file)
                label = 1 if 'real' in subdir else 0
                try:
                    data, sample_rate = librosa.load(filepath)
                    feature = feature_extractor(data, sample_rate)
                    feature_array.append(feature)
                except Exception as e:
                    print(f"Error processing file {filepath}: {e}")
    
    feature_array = [pad_array(feature, max_rows, max_cols) for feature in feature_array]
    feature_array = np.array(feature_array)
    feature_array = feature_array[..., np.newaxis]
    feature_array = np.expand_dims(feature_array, axis=1)

    return feature_array

feature_extractors = {
    'tonnetz': extract_tonnetz_features,
    'mfcc': extract_mfcc_features,
    'chroma_stft': extract_chroma_stft_features,
    'spectral_centroid': extract_spectral_centroid_features,
    'spectral_contrast': extract_spectral_contrast_features
}

X_all_cnn, y_cnn, size_cnn, files_to_predict_all_cnn, file_labels_cnn = generate_data_CNN(audio_files_directory, feature_extractors)

predicted_labels_average = np.zeros(len(file_labels_cnn))
predicted_labels_weighted_average = np.zeros(len(file_labels_cnn))
models = {}
accuracy_scores = {}
for key, X in X_all_cnn.items():
    models[key], accuracy_scores[key] = cnn_lstm_model(X, y_cnn, size_cnn, 0.6)

weight_sum = sum(accuracy_scores.values())
for key, file in files_to_predict_all_cnn.items():
    predicted = models[key].predict(file)
    
    for i in range(len(predicted)):
        predicted_labels_average[i] += predicted[i][0] / len(files_to_predict_all_cnn.keys())
        predicted_labels_weighted_average[i] += ((predicted[i][0] * accuracy_scores[key]) / weight_sum)

for prediction_avg, prediction_weighted_avg, label in zip(predicted_labels_average, predicted_labels_weighted_average, file_labels_cnn):
    print(f"Predicted Average: {prediction_avg:.4f}, Predicted Weighted Average: {prediction_weighted_avg:.4f}, Actual: {label}")


lstm_models = {}
accuracy_scores = {}

X_all_lstm, y_lstm, test_files_lstm, test_labels_lstm = generate_data_LSTM(audio_files_directory, feature_extractors)

for key, X in X_all_lstm.items():
    lstm_models[key], accuracy_scores[key] = lstm_model(X, y_lstm, 0.6)

weight_sum_lstm = sum(accuracy_scores.values())
file_predictions = {}
number_of_chunks = {}
file_labels = {}
prediction_choice = 0
for key, (file_data, filename) in test_files_lstm.items():
    predicted = lstm_models[key].predict(file_data)

    for i in range(len(predicted)):
        if predicted[i] >= 0.7:
            prediction_choice = 1
        elif predicted[i] <= 0.4:
            prediction_choice = 0
        else:
            prediction_choice = 0.5
        if filename[i] not in file_predictions:
            file_predictions[filename[i]] = prediction_choice
            file_labels[filename[i]] = test_labels_lstm[i]
        else:
            file_predictions[filename[i]] += prediction_choice

        if filename[i] not in number_of_chunks:
            number_of_chunks[filename[i]] = 1
        else:
            number_of_chunks[filename[i]] += 1
    
for filename, prediction in file_predictions.items():
    file_predictions[filename] = prediction / number_of_chunks[filename]

end_time = time.time()

for prediction_avg, label in zip(file_predictions.values(), test_labels_lstm):
    print(f"Predicted: {prediction_avg:.4f}, Actual: {label}")

print('Time taken:', end_time - start_time)

print("prediction size")
print(len(file_predictions))
print("separate prediction from read labels")
print(len(test_labels_lstm))
print("separate prediction from read labels")
print(len(file_labels))
print("prediction size")
# Plotting the predictions
plt.figure(figsize=(15, 7))

plt.plot(file_predictions.values(), label='Predicted', marker='o')
plt.plot(file_labels.values(), label='Actual Labels', marker='s')
plt.plot(predicted_labels_weighted_average, label='Predicted Weighted Average', marker='x')

plt.title('Predicted vs Actual Labels')
plt.xlabel('Sample Index')
plt.ylabel('Label')
plt.legend()
plt.show()