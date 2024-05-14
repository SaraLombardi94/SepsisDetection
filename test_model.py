# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 08:45:26 2024

@author: Utente
"""

import numpy as np
import os
from glob import glob
import tensorflow.keras
import tensorflow as tf
import random
import sklearn
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.metrics import accuracy_score



DATASETDIR = r'C:\Users\Utente\Desktop\wetransfer_controls-microcirculation_2024-04-23_1250\controls-microcirculation\test_tf'
MODELDIR = r'C:\Users\Utente\Desktop\wetransfer_controls-microcirculation_2024-04-23_1250\controls-microcirculation\tf_bilanciato\modelli'
MODELNAME = '5fold_dsTM2min30_gru_mse_bs32_lre0.0001_windows3750onset_jitterFalse_ep150_1'
LOGDIR = os.path.join(MODELDIR,MODELNAME,'logs')
WEIGHTSDIR = os.path.join(MODELDIR,MODELNAME,'weights')
FS = 125
CLASSES = ['control','microcirculation']
NORMALIZE = True
USE_WINDOWS = True
NORMRANGE = (-1,1)
WINDOW_LENGTH = FS * 30 * 1
BATCH_SIZE = 16

def normalize(x):
   x = sklearn.preprocessing.minmax_scale(x, feature_range = NORMRANGE)
   return x

def createLabels(data,classes):
  labels = []
  for filepath in data:
    group = filepath.split(os.path.sep)[-2]
    if group in CLASSES:
      class_num = CLASSES.index(group)
      labels.append(class_num)
  return labels


def fix_shape(x, y):
  length = WINDOW_LENGTH
  x.set_shape([None, length, 1])
  y.set_shape([None, 2])
  return x, y


def load_and_select_window(filepath, y):
    # Assicurati che filepath sia una stringa
    filepath = tf.compat.as_str_any(filepath)
    # Costruisci il percorso per i punti di inizio basato sul percorso del file originale
    pathToPoints = filepath.removesuffix('.npz') + '.txt'  # Assumi che i punti di inizio siano salvati in un file .txt con lo stesso nome base
    # Carica l'elenco dei punti di inizio
    onsetList = np.loadtxt(pathToPoints).astype(np.int64)
    start_timestep = random.choice(onsetList)
    # Carica i dati del segnale
    signal_data = np.load(filepath)['arr_0'].astype('float64')
    signal_data = np.reshape(signal_data, [signal_data.size, 1])
    if NORMALIZE:
        signal_data = normalize(signal_data)
    if USE_WINDOWS:
        while (signal_data[start_timestep:]).size < WINDOW_LENGTH:
            start_timestep = random.choice(onsetList)
        signal_data = signal_data[start_timestep:start_timestep + WINDOW_LENGTH]
    y = to_categorical(y, num_classes=len(CLASSES))
    return signal_data, y

def create_dataset(x_test, y_test):
  ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
  # Train dataset
  ds_test = ds_test.map(lambda filepath, label : tf.numpy_function(
        load_and_select_window, [filepath, label], (tf.double, tf.float32)))
  ds_test = ds_test.cache()
  ds_test = ds_test.batch(BATCH_SIZE)
  ds_test = ds_test.map(fix_shape)
  return ds_test



dataPaths = glob(os.path.join(f'{DATASETDIR}','control','*.npz'))+ glob(os.path.join(f'{DATASETDIR}','microcirculation','*.npz'))

labels = createLabels(dataPaths,CLASSES)

ds_test = create_dataset(dataPaths,labels)


if USE_WINDOWS:
  INPUT_SHAPE = (WINDOW_LENGTH,1)


# Sostituisci `i` con il numero della fold che vuoi caricare, e inserisci il percorso corretto
fold_number = 0  # Sostituisci `i` con il numero di fold che ti interessa
model_path = os.path.join(WEIGHTSDIR, f'{fold_number}fold.keras')

# Carica il modello
model = tf.keras.models.load_model(model_path)

#faccio le previsioni
predictions = model.predict(ds_test)
predicted_classes = np.argmax(predictions, axis=1)

#ottengo le etichette vere
true_classes = np.array(labels)

# Calcolo della precisione con scikit-learn
accuracy = accuracy_score(true_classes, predicted_classes)
print(f"Precisione del modello: {accuracy * 100:.2f}%")



#confronto le labels vere con le predette
correct_predictions = predicted_classes == true_classes
incorrect_predictions = np.invert(correct_predictions)

#stampo l'indice dei segnali predetti corretti  
print("Indici delle predizioni corrette:", np.where(correct_predictions)[0])

# stampo l'indice dei segnali predetti sbagliati e il loro nome  
incorrect_indices = np.where(incorrect_predictions)[0]
print("Indici delle predizioni errate:", incorrect_indices)

# Stampa dei nomi dei file per le predizioni errate
incorrect_files = np.array(dataPaths)[incorrect_indices]
for index, file in zip(incorrect_indices, incorrect_files):
    print(f"Indice errato: {index}, File: {file}, Etichetta predetta: {predicted_classes[index]}, Etichetta vera: {true_classes[index]}")

