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

class TFSpectralConvergence(tf.keras.layers.Layer):
    """Spectral convergence loss."""

    def __init__(self):
        """Initialize."""
        super().__init__()

    def __call__(self, y_mag, x_mag):
        """Calculate forward propagation.
        Args:
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        return tf.norm(y_mag - x_mag, ord="fro", axis=(-2, -1)) / tf.norm(
            y_mag, ord="fro", axis=(-2, -1)
        )


class TFLogSTFTMagnitude:

    def __call__(self, y_mag, x_mag):
        return tf.abs(tf.math.log(y_mag) - tf.math.log(x_mag))


class TFSTFT:

    def __init__(self, frame_length=600, frame_step=120, fft_length=1024):
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_length = fft_length
        self.spectral_convergenge_loss = TFSpectralConvergence()
        self.log_stft_magnitude_loss = TFLogSTFTMagnitude()

    def __call__(self, y, x):
        x_mag = tf.abs(
            tf.signal.stft(
                signals=x,
                frame_length=self.frame_length,
                frame_step=self.frame_step,
                fft_length=self.fft_length,
            )
        )
        y_mag = tf.abs(
            tf.signal.stft(
                signals=y,
                frame_length=self.frame_length,
                frame_step=self.frame_step,
                fft_length=self.fft_length,
            )
        )

        x_mag = tf.clip_by_value(tf.math.sqrt(x_mag ** 2 + 1e-7), 1e-7, 1e3)
        y_mag = tf.clip_by_value(tf.math.sqrt(y_mag ** 2 + 1e-7), 1e-7, 1e3)

        sc_loss = self.spectral_convergenge_loss(y_mag, x_mag)
        mag_loss = self.log_stft_magnitude_loss(y_mag, x_mag)

        return sc_loss, mag_loss


def spectral_convergence_loss(y, x):

    fft_lengths=[1024, 2048, 512]
    frame_lengths=[600, 1200, 240]
    frame_steps=[120, 240, 50]
    assert len(frame_lengths) == len(frame_steps) == len(fft_lengths)
    stft_losses = []
    for frame_length, frame_step, fft_length in zip(
        frame_lengths, frame_steps, fft_lengths
    ):
        stft_losses.append(TFSTFT(frame_length, frame_step, fft_length))
    sc_loss = 0.0
    mag_loss = 0.0
    for f in stft_losses:
        sc_l, mag_l = f(y, x)
        sc_loss += tf.reduce_mean(sc_l, axis=list(range(1, len(sc_l.shape))))
        mag_loss += tf.reduce_mean(mag_l, axis=list(range(1, len(mag_l.shape))))

    sc_loss /= len(stft_losses)
    mag_loss /= len(stft_losses)

    return (sc_loss + mag_loss) / 2


def custom_loss(y, x):
    print('loss y shape', y.shape, flush=True)
    print('loss x shape', x.shape, flush=True)
    mse = tf.keras.losses.MeanSquaredError()
    mse_res = mse(y, x)
    print('mse', mse_res, flush=True)

    sp_res = tf.reduce_mean(spectral_convergence_loss(tf.squeeze(y, -1), tf.squeeze(x, -1)))
    print('sp_res', sp_res, flush=True)
    return mse_res + sp_res
    
# Define the paths to the folders containing the noisy and clean signals
noisy_signals_folder = "/NFSHOME/vlarikova/noisy"
clean_signals_folder = "/NFSHOME/vlarikova/clean"

# Load the signals from the folders
noisy_signals = []
clean_signals = []

for filename in os.listdir(noisy_signals_folder):
    if filename.endswith(".wav"):
        noisy_signal, sr = sf.read(os.path.join(noisy_signals_folder, filename))
        clean_signal, sr = sf.read(os.path.join(clean_signals_folder, filename))
        noisy_signals.append((noisy_signal, filename))
        clean_signals.append((clean_signal, filename))

#slice_len = 16384
slice_len = 131072

# min_len = 10000000000000000
# for s in clean_signals:
    # min_len = min(min_len, len(s))
# for s in noisy_signals:
    # min_len = min(min_len, len(s))

# Split the data into training, validation, and test sets
noisy_train, noisy_valtest, clean_train, clean_valtest = train_test_split(noisy_signals, clean_signals, test_size=0.2, random_state=42)
noisy_val, noisy_test, clean_val, clean_test = train_test_split(noisy_valtest, clean_valtest, test_size=0.5, random_state=42)

padded_noisy_signals_train = []
padded_clean_signals_train = []

for (noisy_signal, noisy_filename), (clean_signal, clean_filename) in zip(noisy_train, clean_train):
    to_pad = ((len(noisy_signal) - 1) // slice_len + 1) * slice_len - len(noisy_signal)
    padded_noisy_signal = np.concatenate((noisy_signal, np.zeros(to_pad)))
    padded_clean_signal = np.concatenate((clean_signal, np.zeros(to_pad)))

    noisy_matrix = []
    clean_matrix = []

    for start in range(0, len(padded_noisy_signal), slice_len):
        noisy_matrix.append(padded_noisy_signal[start:start+slice_len])
        clean_matrix.append(padded_clean_signal[start:start+slice_len])

    for n, c in zip(noisy_matrix, clean_matrix):
        padded_noisy_signals_train.append(n)
        padded_clean_signals_train.append(c)

    # padded_noisy_signals_train.append(noisy_signal[:min_len])
    # padded_clean_signals_train.append(clean_signal[:min_len])

padded_noisy_signals_val = []
padded_clean_signals_val = []

for (noisy_signal, noisy_filename), (clean_signal, clean_filename) in zip(noisy_val, clean_val):
    
    to_pad = ((len(noisy_signal) - 1) // slice_len + 1) * slice_len - len(noisy_signal)
    padded_noisy_signal = np.concatenate((noisy_signal, np.zeros(to_pad)))
    padded_clean_signal = np.concatenate((clean_signal, np.zeros(to_pad)))

    noisy_matrix = []
    clean_matrix = []

    for start in range(0, len(padded_noisy_signal), slice_len):
        noisy_matrix.append(padded_noisy_signal[start:start+slice_len])
        clean_matrix.append(padded_clean_signal[start:start+slice_len])

    for n, c in zip(noisy_matrix, clean_matrix):
        padded_noisy_signals_val.append(n)
        padded_clean_signals_val.append(c)


    # padded_noisy_signals_val.append(noisy_signal[:min_len])
    # padded_clean_signals_val.append(clean_signal[:min_len])

padded_noisy_signals_test = []
padded_clean_signals_test = []

for (noisy_signal, noisy_filename), (clean_signal, clean_filename) in zip(noisy_test, clean_test):

    to_pad = ((len(noisy_signal) - 1) // slice_len + 1) * slice_len - len(noisy_signal)
    padded_noisy_signal = np.concatenate((noisy_signal, np.zeros(to_pad)))
    padded_clean_signal = np.concatenate((clean_signal, np.zeros(to_pad)))

    noisy_matrix = []
    clean_matrix = []

    for start in range(0, len(padded_noisy_signal), slice_len):
        noisy_matrix.append(padded_noisy_signal[start:start+slice_len])
        clean_matrix.append(padded_clean_signal[start:start+slice_len])

    for n, c in zip(noisy_matrix, clean_matrix):
        padded_noisy_signals_test.append(n)
        padded_clean_signals_test.append(c)
    # padded_noisy_signals_test.append(noisy_signal[:min_len])
    # padded_clean_signals_test.append(clean_signal[:min_len])

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

# Input shape for the model
input_shape = (slice_len, 1)

# Create the model with 6 blocks and 128 filters per block
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

output_folder = "/NFSHOME/kvolkov/Lera/one_noise_mse"
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