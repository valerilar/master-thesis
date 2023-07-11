import os
import math
import glob
import scipy
import pickle
import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from numpy.lib import stride_tricks
from tensorflow.keras import models, layers 
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model

def do_stft(signal):
    
    window_fn = tf.signal.hamming_window

    win_size=1024
    hop_size=512

    stft_signal=tf.signal.stft(signal,frame_length=win_size, window_fn=window_fn, frame_step=hop_size, pad_end=True)
    stft_stacked=tf.stack( values=[tf.math.real(stft_signal), tf.math.imag(stft_signal)], axis=-1)

    return stft_stacked

def do_istft(data):
    
    window_fn = tf.signal.hamming_window

    win_size=1024
    hop_size=512

    inv_window_fn=tf.signal.inverse_stft_window_fn(hop_size, forward_window_fn=window_fn)

    pred_cpx=tf.complex(data[...,0], data[...,1])
    pred_time=tf.signal.inverse_stft(pred_cpx, win_size, hop_size, window_fn=inv_window_fn)
    return pred_time

def data_generator(noisy_signals, clean_signals, min_len, batch_size=32):
    while True:
        noisy_batch = []
        clean_batch = []
        for _ in range(batch_size):
            index = np.random.randint(len(noisy_signals))
            noisy_signal, sr = sf.read(noisy_signals[index][0])
            clean_signal, sr = sf.read(clean_signals[index][0])
            noisy_spec = do_stft(noisy_signal[:min_len])
            clean_spec = do_stft(clean_signal[:min_len])
            noisy_batch.append(noisy_spec)
            clean_batch.append(clean_spec)
        yield np.array(noisy_batch), np.array(clean_batch)
        
# Define the paths to the folders containing the noisy and clean signals
noisy_signals_folder = "/NFSHOME/vlarikova/noisy"
clean_signals_folder = "/NFSHOME/vlarikova/clean"

# Load the signals from the folders
noisy_signals = []
clean_signals = []

for filename in os.listdir(noisy_signals_folder):
    if filename.endswith(".wav"):
        noisy_signals.append((os.path.join(noisy_signals_folder, filename), filename))
        clean_signals.append((os.path.join(clean_signals_folder, filename), filename))

# Determine the minimum length of signals
min_len = min(len(sf.read(s[0])[0]) for s in clean_signals + noisy_signals)

# Split the data into training, validation, and test sets
noisy_train, noisy_valtest, clean_train, clean_valtest = train_test_split(noisy_signals, clean_signals, test_size=0.2, random_state=42)
noisy_val, noisy_test, clean_val, clean_test = train_test_split(noisy_valtest, clean_valtest, test_size=0.5, random_state=42)

sr = 0

# Apply STFT on the training set
def apply_stft(noisy_signals, clean_signals, min_len):
    X = []
    y = []

    global sr

    for i in range(len(noisy_signals)):
        noisy_signal, freq = sf.read(noisy_signals[i][0])
        clean_signal, freq = sf.read(clean_signals[i][0])
        sr = freq
        noisy_spec = do_stft(noisy_signal[:min_len])
        clean_spec = do_stft(clean_signal[:min_len])
        X.append(noisy_spec)
        y.append(clean_spec)
        
    X = np.array(X)
    y = np.array(y)
    
    X = np.expand_dims(X, axis=-1)
    y = np.expand_dims(y, axis=-1)
    
    return X, y

X_train, y_train = apply_stft(noisy_train, clean_train, min_len)
X_val, y_val = apply_stft(noisy_val, clean_val, min_len)
X_test, y_test = apply_stft(noisy_test, clean_test, min_len)

# Print the shapes of the spectrogram data
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

def unet_denoiser(input_shape):
    inputs = layers.Input(shape=input_shape)

    x = inputs

    # Encoder path

    num_blocks = 4
    skips = []

    chin_start = 2
    filter_start = 64

    chins = []
    filters = []

    output_shapes = []

    for i in range(num_blocks):
        x = layers.Conv2D(filter_start, 3, activation='relu', padding='same')(x)
        x = layers.Conv2D(filter_start, 3, activation='relu', padding='same')(x)
        output_shapes.append(x.shape)
        x = layers.MaxPooling2D(pool_size=(2, 2), padding='valid')(x)
        skips.append(x)
        chins.append(chin_start)
        filters.append(filter_start)
        if i < num_blocks - 1:
            chin_start = filter_start
            filter_start *= 2

    x = layers.Dense(filter_start, activation='relu')(x)
    x = layers.Dense(filter_start, activation='relu')(x)

    chouts = chins[::-1]
    filters_r = filters[::-1]

    for i in range(num_blocks):
        x = layers.Add()([x, skips[num_blocks - i - 1]])
        x = layers.Conv2D(filters_r[i], 3, activation='relu', padding='same')(x)
        activation = 'relu' if i < num_blocks - 1 else None
        x = layers.Conv2DTranspose(chouts[i], 3, strides=(2, 2), padding='valid', activation=activation)(x)
        x = layers.Cropping2D(((0, x.shape[1] - output_shapes[-i-1][1]), (0, x.shape[2] - output_shapes[-i-1][2])))(x)
    model = models.Model(inputs=inputs, outputs=x)
    return model

# Define the input shape
input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])

# Create the U-Net denoiser model
model = unet_denoiser(input_shape)

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss='mean_squared_logarithmic_error')
model.summary()

# Define early stopping criteria
early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10, restore_best_weights=True)

model_save = ModelCheckpoint('speech_denoising_model_best.h5', save_best_only=True, monitor='val_loss', mode='min')

# Train the model with early stopping
num_epochs = 500
batch_size = 32

steps_per_epoch = len(noisy_train) // batch_size

history = model.fit(data_generator(noisy_train, clean_train, min_len, batch_size=batch_size),
                    epochs=num_epochs,
                    steps_per_epoch=steps_per_epoch,
                    verbose=1,
                    callbacks=[early_stopping, model_save],
                    validation_data=(X_val, y_val))

# Plot the training loss and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

output_folder = "/NFSHOME/kvolkov/Lera/output_conv_spec/signals_2"
plot_path = os.path.join(output_folder, 'loss_plot.png')
plt.savefig(plot_path)
plt.close()

# Save the model
model.save(os.path.join(output_folder, 'speech_denoising_model.h5'))

# Denoise the spectrograms in the test set
denoised_signals_test = model.predict(X_test)

denoised_signals_time = []

for i, denoised_spec in enumerate(denoised_signals_test):
    # Apply inverse Short-Time Fourier Transform (iSTFT)
    denoised_signal = do_istft(denoised_spec)

    # Remove zero-padding at the end
    denoised_signal = denoised_signal[:min_len]

    denoised_signals_time.append(denoised_signal)

for denoised_signal, info in zip(denoised_signals_time, clean_test):
    denoised_signal_path = os.path.join(output_folder, 'clean_signal_test_{}'.format(info[1]))
    sf.write(denoised_signal_path, denoised_signal, sr)
