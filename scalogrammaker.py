# -*- coding: utf-8 -*-
"""
Created on Tue May 14 09:56:25 2024

@author: Utente
"""

import os 
import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
import tensorflow as tf
import ssqueezepy as sp
#from ssqueezepy.visuals import plot, imshow
from scipy.signal import cwt, morlet
from PIL import Image
import io
import random
import sklearn
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold, train_test_split, GroupShuffleSplit, StratifiedGroupKFold, GroupKFold
from sklearn import preprocessing
from sklearn.utils import shuffle

f = 125 #frequenza di campionamento
t_sample = 1/f #tempo di campionamento
n_samples = f*30 #numero di campioni considerato del segnale discreto
NORMALIZE = True
NORMRANGE = (-1,1)
USE_WINDOWS = True
USE_SCALOGRAM = True
WINDOW_LENGTH = 125 * 30 * 1
CLASSES = ['control', 'microcirculation']

input_cartella = r'C:\Users\Utente\Desktop\minidataset\control'
output_cartella = r'C:\Users\Utente\Desktop\wetransfer_controls-microcirculation_2024-04-23_1250\controls-microcirculation\scalogrammi\control'

# Definizione delle scale per la CWT
#custom_scales = np.arange(41, 125)#○,3159,dtype='float64')  # Gli intervalli di scale possono essere adattati a seconda delle necessità



'''funzione che rimuove la componente continua a 0Hz del segnale'''
def sottrai_media(lista):
    mean_value = np.mean(lista)
    nuova_lista = np.array([valore - mean_value for valore in lista])
    return nuova_lista

def compute_scalogram(x,y):
    tf.print("ok")
    # Calcola lo scalogramma
    Wx, scales, _ = sp.cwt(x, 'morlet', scales='log', derivative=False, nv=32)
    Wx = np.abs(Wx)
    return Wx,y 
        

def convert_to_image_and_label(Wx, label):
    Wx_with_channel = Wx[:, :, np.newaxis]
    # Duplica i canali per creare un'immagine RGB
    Wx_rgb = np.concatenate([Wx_with_channel]*3, axis=-1)
    tf.print(f"{Wx_rgb.shape}_conv")
    
    # Converti l'array in un'immagine PIL
    image_rgb = tf.keras.utils.array_to_img(Wx_rgb)
    
    # Converti l'immagine PIL in un Tensor TensorFlow
    #image_tensor = tf.convert_to_tensor(np.array(image_rgb), dtype=tf.float32)
    #tf.print(f"{image_tensor.shape}_conv")
    return image_rgb, label



def normalize(x):
    x = sklearn.preprocessing.minmax_scale(x, feature_range=NORMRANGE)
    return x



        
def load_and_select_window_with_scalogram(filepath, y):
    filepath = tf.compat.as_str_any(filepath)
    pathToPoints = filepath.removesuffix('.npz') + '.txt'
    onsetList = np.loadtxt(pathToPoints).astype(np.int64)
    start_timestep = random.choice(onsetList)
    signal_data = np.load(filepath)['arr_0'].astype('float64')
    #signal_data = np.reshape(signal_data, [signal_data.size, 1])
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
    #image_tensor = tf.convert_to_tensor(np.array(image_rgb), dtype=tf.float32)
    return image_rgb, y     

    
img, classe = load_and_select_window_with_scalogram(r'C:/Users/Utente/Desktop/minidataset/control/plre121.npy_#268.npz',0)

plt.figure(figsize=(10, 6))
# plt.subplot(1, 2, 1)
# plt.title('Scalogramma (Wx_with_channels)')
# plt.imshow(Wx_with_channels[:, :, 0], aspect='auto',cmap='gray')
  
plt.subplot(1, 2, 2)
plt.title('Scalogramma RGB (Wx_rgb)')
plt.imshow(img, aspect='auto')
plt.show()






# Itera attraverso i percorsi dei file NPZ
for nome_file_npz in os.listdir(input_cartella):
    if 'npz' in nome_file_npz:
        
        
        
        
        percorso_file_npz = os.path.join(input_cartella, nome_file_npz)
        # Carica il file NPZ
        dati_npz = np.load(percorso_file_npz, allow_pickle=True)
        dati_npz.files
        sig = dati_npz['arr_0'] #trasformo il segnale in una lista di valori
        sig = sig[:n_samples]
        tf.print(f'{sig.shape}_conv')
        Wx, label = compute_scalogram(sig,y=1)
        
        
        Wx_rgb , label =convert_to_image_and_label(Wx, label)
        
        #Wx_rgb = (Wx_rgb - Wx_rgb.min()) / (Wx_rgb.max() - Wx_rgb.min())
        
        plt.figure(figsize=(10, 6))
        # plt.subplot(1, 2, 1)
        # plt.title('Scalogramma (Wx_with_channels)')
        # plt.imshow(Wx_with_channels[:, :, 0], aspect='auto',cmap='gray')
  
        plt.subplot(1, 2, 2)
        plt.title('Scalogramma RGB (Wx_rgb)')
        plt.imshow(Wx_rgb, aspect='auto')
        plt.show()
