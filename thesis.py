import os
from scipy.io import wavfile
import numpy as np
from sklearn.preprocessing import StandardScaler
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

real_dataset_directory = 'C:/Users/giga-/OneDrive/Desktop/Thesis/FakeOrReal dataset/for-original/training/real'
fake_dataset_directory = 'C:/Users/giga-/OneDrive/Desktop/Thesis/FakeOrReal dataset/for-original/training/fake'
real_files_content = read_wav_files(real_dataset_directory)
fake_files_content = read_wav_files(fake_dataset_directory)

for filename, content in real_files_content.items():
    print(f'Filename: {filename}')
    print(f'Sample Rate: {content["sample_rate"]} Hz')
    print(f'Data Shape: {content["data"].shape}')
    print(f'Data Type: {content["data"].dtype}')
    print('First few samples:', content["data"][:20])
    print('-' * 50 + '\n')

# Initialize scaler
scaler = StandardScaler()

# Process each audio file
processed_data = {}
for filename, content in real_files_content.items():
    # Convert to mono if stereo by taking mean of channels
    if len(content['data'].shape) > 1:
        audio_data = np.mean(content['data'], axis=1)
    else:
        audio_data = content['data']
    
    # Normalize the data
    audio_data = audio_data.reshape(-1, 1)
    audio_data_scaled = scaler.fit_transform(audio_data)
    
    # Segment into fixed-length sequences (e.g., 1000 samples)
    sequence_length = 1000
    sequences = []
    for i in range(0, len(audio_data_scaled) - sequence_length, sequence_length):
        sequence = audio_data_scaled[i:i + sequence_length]
        sequences.append(sequence)
    
    processed_data[filename] = np.array(sequences)

for filename, data in processed_data.items():
    print(f'Filename: {filename}')
    print(f'Data Shape: {data.shape}')
    print(f'Data Type: {data.dtype}')
    print('First few sequences:', data[:20])
    print('-' * 50 + '\n')

# Now processed_data contains normalized, segmented sequences ready for LSTM
# Prepare data for LSTM
X = []
y = []

# Add real audio sequences (label 1)
for sequences in processed_data.values():
    X.extend(sequences)
    y.extend([1] * len(sequences))

# Add fake audio sequences (label 0) - process fake data the same way as real
for filename, content in fake_files_content.items():
    if len(content['data'].shape) > 1:
        audio_data = np.mean(content['data'], axis=1)
    else:
        audio_data = content['data']
    
    audio_data = audio_data.reshape(-1, 1)
    audio_data_scaled = scaler.fit_transform(audio_data)
    
    sequences = []
    for i in range(0, len(audio_data_scaled) - sequence_length, sequence_length):
        sequence = audio_data_scaled[i:i + sequence_length]
        sequences.append(sequence)
    X.extend(sequences)
    y.extend([0] * len(sequences))

X = np.array(X)
y = np.array(y)

# Create LSTM model with a simpler architecture
model = Sequential([
    LSTM(32, input_shape=(sequence_length, 1)),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load and process validation data from different directories
val_real_dir = 'C:/Users/giga-/OneDrive/Desktop/Thesis/FakeOrReal dataset/for-original/validation/real'
val_fake_dir = 'C:/Users/giga-/OneDrive/Desktop/Thesis/FakeOrReal dataset/for-original/validation/fake'

val_real_files = read_wav_files(val_real_dir)
val_fake_files = read_wav_files(val_fake_dir)

# Prepare validation data
X_val = []
y_val = []

# Process validation real files
for content in val_real_files.values():
    audio_data = np.mean(content['data'], axis=1) if len(content['data'].shape) > 1 else content['data']
    audio_data = audio_data.reshape(-1, 1)
    audio_data_scaled = scaler.transform(audio_data)
    
    for i in range(0, len(audio_data_scaled) - sequence_length, sequence_length):
        X_val.append(audio_data_scaled[i:i + sequence_length])
        y_val.append(1)

# Process validation fake files
for content in val_fake_files.values():
    audio_data = np.mean(content['data'], axis=1) if len(content['data'].shape) > 1 else content['data']
    audio_data = audio_data.reshape(-1, 1)
    audio_data_scaled = scaler.transform(audio_data)
    
    for i in range(0, len(audio_data_scaled) - sequence_length, sequence_length):
        X_val.append(audio_data_scaled[i:i + sequence_length])
        y_val.append(0)

X_val = np.array(X_val)
y_val = np.array(y_val)

# Train model with separate validation data
history = model.fit(X, y, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Print training results
print("\nTraining Results:")
print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"Final training loss: {history.history['loss'][-1]:.4f}")
print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")


real_test_dir = 'C:/Users/giga-/OneDrive/Desktop/Thesis/FakeOrReal dataset/for-original/testing/real'
fake_test_dir = 'C:/Users/giga-/OneDrive/Desktop/Thesis/FakeOrReal dataset/for-original/testing/fake'
real_test_files = read_wav_files(real_test_dir)
fake_test_files = read_wav_files(fake_test_dir)

# Prepare test data
X_test = []
for content in real_test_files.values():
    audio_data = np.mean(content['data'], axis=1) if len(content['data'].shape) > 1 else content['data']
    audio_data = audio_data.reshape(-1, 1)
    audio_data_scaled = scaler.transform(audio_data)
    
    for i in range(0, len(audio_data_scaled) - sequence_length, sequence_length):
        X_test.append(audio_data_scaled[i:i + sequence_length])

X_test = np.array(X_test)

# Get predictions
predictions = model.predict(X_test)

print('Predictions:', predictions)
print('-' * 50 + '\n')

# Prepare test data
X_test = []
for content in fake_test_files.values():
    audio_data = np.mean(content['data'], axis=1) if len(content['data'].shape) > 1 else content['data']
    audio_data = audio_data.reshape(-1, 1)
    audio_data_scaled = scaler.transform(audio_data)
    
    for i in range(0, len(audio_data_scaled) - sequence_length, sequence_length):
        X_test.append(audio_data_scaled[i:i + sequence_length])

X_test = np.array(X_test)

# Get predictions
predictions = model.predict(X_test)


print('Predictions:', predictions)
print('-' * 50 + '\n')

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()