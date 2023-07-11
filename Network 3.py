import os
import numpy as np
import soundfile as sf
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import BatchNormalization
from keras.layers import Dropout
from keras import regularizers
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Activation, Add, Conv1D, Dense, Multiply
from tensorflow.keras.models import Model

def build_first_model(input_shape,
                num_convs=4,
                hidden_start=48,
                kernel_size=8,
                stride=4):
    input_tensor = layers.Input(shape=input_shape)
    x = input_tensor

    skip_connections = []

    chin = 1
    chout = 1

    hiddens = [hidden_start]
    for i in range(1, num_convs):
        hiddens.append(hiddens[i - 1] * 2)

    chins = [1]
    for i in range(1, num_convs):
        chins.append(hiddens[i - 1])

    hiddens_out = hiddens[::-1]
    chouts = chins[::-1]

    for i in range(num_convs):
        x = layers.Conv1D(chins[i], kernel_size=kernel_size, padding='same', activation ='relu')(x)
        x = layers.Conv1D(hiddens[i], kernel_size=1, strides=stride, padding='same', activation='relu')(x)
        skip_connections.append(x)

    x = layers.LSTM(hiddens[-1], activation='relu', return_sequences=True)(x)
    x = layers.LSTM(hiddens[-1], activation='relu', return_sequences=True)(x)

    chout = chin
    for i in range(num_convs):
        x = layers.Add()([x, skip_connections[num_convs - i - 1]])
        x = layers.Conv1D(hiddens_out[i], kernel_size=1, padding='same', activation='relu')(x)
        activation = 'relu' if i < num_convs - 1 else None
        x = layers.Conv1DTranspose(chouts[i], kernel_size, stride, padding='same', activation=activation)(x)

    model = models.Model(inputs=input_tensor, outputs=x)
    return model

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

# Define the paths to the folders containing the noisy and clean signals
noisy_signals_folder = "/NFSHOME/kvolkov/Lera/noisy"
clean_signals_folder = "/NFSHOME/kvolkov/Lera/clean"

# Load the signals from the folders
noisy_signals = []
clean_signals = []

slice_len = 131072

# Input shape for the WaveNet model
input_shape = (slice_len, 1)
model_1 = build_first_model(input_shape)
model_1.load_weights('/NFSHOME/kvolkov/Lera/output_new_loss_noises/speech_denoising_model.h5')

min_len = 160001

model_2 = unet_denoiser((313, 513, 2))
model_2.load_weights('/NFSHOME/kvolkov/Lera/output_conv_spec/signals_4/speech_denoising_model.h5')


def split_s(signal, slice_len):
    to_pad = ((len(noisy_signal) - 1) // slice_len + 1) * slice_len - len(noisy_signal)
    
    padded_shape = list(signal.shape)
    padded_shape[0] += to_pad

    padded_signal = np.zeros(padded_shape)
    padded_signal[:len(signal)] = signal

    matrix = []

    for start in range(0, len(padded_signal), slice_len):
        matrix.append(padded_signal[start:start+slice_len])

    return np.array(matrix)

def merge_s(signals, full_len):
    return signals.flatten()[:full_len]


raw_signals = []
for filename in os.listdir(noisy_signals_folder):
    if filename.endswith(".wav"):
        noisy_signal, sr = sf.read(os.path.join(noisy_signals_folder, filename))


        noisy_signal = noisy_signal[:min_len]
        raw_signals.append(noisy_signal)

        clean_signal, sr = sf.read(os.path.join(clean_signals_folder, filename))
        clean_signal = clean_signal[:min_len]
        clean_signals.append((clean_signal, filename))

raw_signals = np.array(raw_signals)

model_1_data = []
for noisy_signal in raw_signals:
    noisy_matrix = split_s(noisy_signal, slice_len)
    model_1_data.append(noisy_matrix)
model_1_data = np.array(model_1_data)
p = model_1.predict(model_1_data)

model_1_clean = []
used_rows = 0
for signal in raw_signals:
    rows = (len(signal) - 1) // slice_len + 1
    denoised = p[used_rows:used_rows+rows].flatten()[:len(signal)]
    model_1_clean.append(denoised, filename)
    used_rows += rows
model_1_clean = np.array(model_1_clean)

stfts = []
for noisy_signal in raw_signals:
    stfts.append(do_stft(noisy_signal))
stfts = np.array(stfts)
p = model_2.predict(stfts)

model_2_clean = []
for stft in p:
    model_2_clean.append(do_istft(stft)[:min_len])
model_2_clean = np.array(model_2_clean)

data = np.array([raw_signals, model_1_clean, model_2_clean])
data = np.swapaxes(data, 0, 2)

# Split the data into training, validation, and test sets
noisy_train, noisy_valtest, clean_train, clean_valtest = train_test_split(noisy_signals, clean_signals, test_size=0.2, random_state=42)
noisy_val, noisy_test, clean_val, clean_test = train_test_split(noisy_valtest, clean_valtest, test_size=0.5, random_state=42)

padded_noisy_signals_train = []
padded_clean_signals_train = []

for noisy_signal, (clean_signal, clean_filename) in zip(noisy_train, clean_train):
    to_pad = ((len(noisy_signal) - 1) // slice_len + 1) * slice_len - len(noisy_signal)

    padded_shape = list(signal.shape)
    padded_shape[0] += to_pad
    padded_noisy_signal = np.zeros(padded_shape)

    padded_clean_signal = np.concatenate((clean_signal, np.zeros(to_pad)))

    noisy_matrix = []
    clean_matrix = []

    for start in range(0, len(padded_noisy_signal), slice_len):
        noisy_matrix.append(padded_noisy_signal[start:start+slice_len])
        clean_matrix.append(padded_clean_signal[start:start+slice_len])

    for n, c in zip(noisy_matrix, clean_matrix):
        padded_noisy_signals_train.append(n)
        padded_clean_signals_train.append(c)

padded_noisy_signals_val = []
padded_clean_signals_val = []

for noisy_signal, (clean_signal, clean_filename) in zip(noisy_val, clean_val):
    
    to_pad = ((len(noisy_signal) - 1) // slice_len + 1) * slice_len - len(noisy_signal)
    padded_shape = list(signal.shape)
    padded_shape[0] += to_pad
    padded_noisy_signal = np.zeros(padded_shape)
    padded_clean_signal = np.concatenate((clean_signal, np.zeros(to_pad)))

    noisy_matrix = []
    clean_matrix = []

    for start in range(0, len(padded_noisy_signal), slice_len):
        noisy_matrix.append(padded_noisy_signal[start:start+slice_len])
        clean_matrix.append(padded_clean_signal[start:start+slice_len])

    for n, c in zip(noisy_matrix, clean_matrix):
        padded_noisy_signals_val.append(n)
        padded_clean_signals_val.append(c)

padded_noisy_signals_test = []
padded_clean_signals_test = []

for noisy_signal, (clean_signal, clean_filename) in zip(noisy_test, clean_test):

    to_pad = ((len(noisy_signal) - 1) // slice_len + 1) * slice_len - len(noisy_signal)
    padded_shape = list(signal.shape)
    padded_shape[0] += to_pad
    padded_noisy_signal = np.zeros(padded_shape)
    padded_clean_signal = np.concatenate((clean_signal, np.zeros(to_pad)))

    noisy_matrix = []
    clean_matrix = []

    for start in range(0, len(padded_noisy_signal), slice_len):
        noisy_matrix.append(padded_noisy_signal[start:start+slice_len])
        clean_matrix.append(padded_clean_signal[start:start+slice_len])

    for n, c in zip(noisy_matrix, clean_matrix):
        padded_noisy_signals_test.append(n)
        padded_clean_signals_test.append(c)

# Convert the lists of padded signals to numpy arrays
padded_noisy_signals_train = np.array(padded_noisy_signals_train)
padded_clean_signals_train = np.array(padded_clean_signals_train)
padded_noisy_signals_val = np.array(padded_noisy_signals_val)
padded_clean_signals_val = np.array(padded_clean_signals_val)
padded_noisy_signals_test = np.array(padded_noisy_signals_test)
padded_clean_signals_test = np.array(padded_clean_signals_test)

# Reshape the signals to match the expected input shape of the model
X_train = np.expand_dims(padded_noisy_signals_train, axis=-1)
y_train = np.expand_dims(padded_clean_signals_train, axis=-1)
X_val = np.expand_dims(padded_noisy_signals_val, axis=-1)
y_val = np.expand_dims(padded_clean_signals_val, axis=-1)
X_test = np.expand_dims(padded_noisy_signals_test, axis=-1)
y_test = np.expand_dims(padded_clean_signals_test, axis=-1)

print('X_train shape:', X_train.shape)
print('X_val shape:', X_val.shape)
print('X_test shape:', X_test.shape)

def build_model(input_shape,
                num_convs=4,
                hidden_start=48,
                kernel_size=8,
                stride=4):
    input_tensor = layers.Input(shape=input_shape)
    x = input_tensor

    skip_connections = []

    hiddens = [hidden_start]
    for i in range(1, num_convs):
        hiddens.append(hiddens[i - 1] * 2)

    chins = [3]
    for i in range(1, num_convs):
        chins.append(hiddens[i - 1])

    hiddens_out = hiddens[::-1]
    chouts = chins[::-1]
    chouts[-1] = 1

    for i in range(num_convs):
        x = layers.Conv1D(chins[i], kernel_size=kernel_size, padding='same', activation ='relu')(x)
        x = layers.Conv1D(hiddens[i], kernel_size=1, strides=stride, padding='same', activation='relu')(x)
        skip_connections.append(x)

    x = layers.LSTM(hiddens[-1], activation='relu', return_sequences=True)(x)
    x = layers.LSTM(hiddens[-1], activation='relu', return_sequences=True)(x)

    for i in range(num_convs):
        x = layers.Add()([x, skip_connections[num_convs - i - 1]])
        x = layers.Conv1D(hiddens_out[i], kernel_size=1, padding='same', activation='relu')(x)
        activation = 'relu'
        x = layers.Conv1DTranspose(chouts[i], kernel_size, stride, padding='same', activation=activation)(x)

    x = layers.Conv1D(1, kernel_size=1, padding='same', activation='linear')(x)

    model = models.Model(inputs=input_tensor, outputs=x)
    return model

# Input shape for the WaveNet model
input_shape = (slice_len, 3)

# Create the WaveNet model with 6 blocks and 128 filters per block
model = build_model(input_shape)

# Compile the model with MSE as the loss function
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss='mean_squared_logarithmic_error')
model.summary()

# Define early stopping criteria
early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-6, patience=10, restore_best_weights=True)

# Train the model with early stopping
num_epochs = 1000
batch_size = 128

history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=1, callbacks=[early_stopping], validation_data=(X_val, y_val))

# Plot the training loss and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

output_folder = "/NFSHOME/kvolkov/Lera/net3"
plot_path = os.path.join(output_folder, 'loss_plot.png')
plt.savefig(plot_path)
plt.close()

# Save the model
model.save(os.path.join(output_folder, 'speech_denoising_model.h5'))

# Denoise the signals in the test set
denoised_signals_test = model.predict(X_test)
denoised_signals_trimmed_test = []

used_rows = 0
for signal, filename in clean_test:
    rows = (len(signal) - 1) // slice_len + 1
    denoised = denoised_signals_test[used_rows:used_rows+rows].flatten()[:len(signal)]
    denoised_signals_trimmed_test.append((denoised, filename))
    used_rows += rows

# Save the denoised signals to a folder
for i, (denoised_signal, filename) in enumerate(denoised_signals_trimmed_test):
    clean_signal_path = os.path.join(output_folder, 'clean_signal_test_{}'.format(filename))
    sf.write(clean_signal_path, denoised_signal.squeeze(), sr)