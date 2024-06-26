# -*- coding: utf-8 -*-
"""
Created on Sun May 19 19:09:24 2024

@author: Utente
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import ssqueezepy as sp
from sklearn.preprocessing import minmax_scale
from tensorflow.keras.utils import to_categorical
import random

# Configurazioni
f = 125  # Frequenza di campionamento
t_sample = 1/f  # Tempo di campionamento
n_samples = f * 30  # Numero di campioni considerato del segnale discreto
NORMALIZE = True
NORMRANGE = (-1, 1)
USE_WINDOWS = True
USE_SCALOGRAM = True
WINDOW_LENGTH = 125 * 30 * 1
CLASSES = ['control', 'microcirculation']

def normalize(x):
    x = minmax_scale(x, feature_range=NORMRANGE)
    return x

def load_and_select_window_with_scalogram(filepath, y):
    filepath = tf.compat.as_str_any(filepath)
    pathToPoints = filepath.removesuffix('.npz') + '.txt'
    onsetList = np.loadtxt(pathToPoints).astype(np.int64)
    start_timestep = random.choice(onsetList)
    signal_data = np.load(filepath)['arr_0'].astype('float64')
    tf.print(f"{signal_data.shape}_inizio")
    
    if NORMALIZE:
        signal_data = normalize(signal_data)
    
    if USE_WINDOWS:
        tf.print(f'{USE_WINDOWS}')
        while (signal_data[start_timestep:]).size < WINDOW_LENGTH:
            start_timestep = random.choice(onsetList)
        signal_data = signal_data[start_timestep:start_timestep + WINDOW_LENGTH]
    
    y = to_categorical(y, num_classes=len(CLASSES))
    
    tf.print(f"{signal_data.shape}_finestrato")
    
    # Compute scalogram
    Wx, scales, _ = sp.cwt(signal_data, 'morlet', scales='log', derivative=False, nv=32)
    Wx = np.abs(Wx)
    
    # Convert to image
    Wx_with_channel = Wx[:, :, np.newaxis]
    Wx_rgb = np.concatenate([Wx_with_channel]*3, axis=-1)
    
    tf.print(f"{Wx_rgb.shape}_conv")
    
    # Converti l'array in un'immagine PIL
    image_rgb = tf.keras.utils.array_to_img(Wx_rgb)
    return image_rgb, y, Wx, scales  # Restituisci Wx e scales per il plot delle sinusoidi

def compute_scalogram(signal, sampling_rate=f):
    Wx, scales, _ = sp.cwt(signal, 'morlet', scales='log', derivative=False, nv=32)
    Wx = np.abs(Wx)
    return Wx, scales

# Esegui la funzione e plotta il risultato
img, classe, Wx, scales = load_and_select_window_with_scalogram(r'C:/Users/Utente/Desktop/minidataset/control/plre121.npy_#268.npz', 0)

# Genera le sinusoidi a diverse frequenze
t = np.linspace(0, WINDOW_LENGTH / f, WINDOW_LENGTH)
frequencies_to_plot = [0.1, 0.5,1, 2, 5, 10, 20, 30, 60]
sinusoids = {freq: np.sin(2 * np.pi * freq * t) for freq in frequencies_to_plot}

# Calcola gli scalogrammi delle sinusoidi
scalograms = {freq: compute_scalogram(sinusoid) for freq, sinusoid in sinusoids.items()}

plt.figure(figsize=(15, 20))

# Plot dello scalogramma del segnale originale
plt.subplot(5, 2, 1)
plt.title('Scalogramma del segnale originale')
plt.imshow(np.abs(Wx), aspect='auto', cmap='gray')
plt.colorbar(label='Amplitude')
plt.ylabel('Scale')

# Plot degli scalogrammi delle sinusoidi
for i, freq in enumerate(frequencies_to_plot):
    plt.subplot(5, 2, i + 2)
    plt.title(f'Scalogramma della sinusoide a {freq} Hz')
    plt.imshow(scalograms[freq][0], aspect='auto', cmap='gray')
    plt.colorbar(label='Amplitude')
    plt.ylabel('Scale')
    if i + 2 >= 8:
        plt.xlabel('Time')

plt.tight_layout()
plt.show()
