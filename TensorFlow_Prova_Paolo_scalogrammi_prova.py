# -*- coding: utf-8 -*-
"""
Created on Fri May 17 08:55:04 2024

@author: Utente
"""

import random
import os
import numpy as np
import tensorflow.keras
import keras.applications
import tensorflow as tf
from glob import glob
#import tensorflow_addons as tfa
import sklearn
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold, train_test_split, GroupShuffleSplit, StratifiedGroupKFold, GroupKFold
from sklearn import preprocessing
from sklearn.utils import shuffle
from scipy.fft import fft
#model
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Add, Dense, BatchNormalization, Activation, Dropout, Flatten
from tensorflow.keras.layers import Convolution1D, ZeroPadding1D, MaxPooling1D, AveragePooling1D
from tensorflow.keras.layers import LSTM, TimeDistributed
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.initializers import GlorotUniform
import ssqueezepy as sp

# constants
FS = 125
N_CLASSES = 2  # control, sepsis
LR = 1e-4
BATCH_SIZE = 32  # aggiornamento dei pesi della rete
EPOCHS = 150
K = 3
NSAMPLES = FS * 30
WINDOW_LENGTH = FS * 30 * 1
NORMRANGE = (-1, 1)
NORMALIZE = True
USE_SHUFFLE = True
USE_WINDOWS = True
USE_JITTER = False
USE_LOSO = False
USE_SCALOGRAM = True
DROPOUT_RATE = 0.2
RANDOM_STATE = 12
BUFFER_SHUFFLING_SIZE = 180
KERNEL_INITIALIZER = 'glorot_uniform'
LOSSFUNCTION = tf.keras.losses.BinaryCrossentropy()
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LR)
MODELNAME = f'{K}fold_dsTM2min30_gru_plus_adjusted_mse_bs{BATCH_SIZE}_lre{LR}_windows{WINDOW_LENGTH}onset_jitter{USE_JITTER}_ep{EPOCHS}_1'

MODELDIR = r'C:\Users\Utente\Desktop\minidataset\modelli'
DATASETDIR = r'C:\Users\Utente\Desktop\minidataset'
LOGDIR = os.path.join(MODELDIR, MODELNAME, 'logs')
WEIGHTSDIR = os.path.join(MODELDIR, MODELNAME, 'weights')
CLASSES = ['control', 'microcirculation']

class LRTensorBoard(TensorBoard):

  def __init__(self, log_dir, **kwargs):
    super().__init__(log_dir, **kwargs)
    self.lr_writer = tf.summary.create_file_writer(self.log_dir + '/learning')
    self.test_writer = tf.summary.create_file_writer(self.log_dir + '/testing')

  def on_epoch_end(self, epoch, logs=None):
    lr = getattr(self.model.optimizer, 'lr', None)
    with self.lr_writer.as_default():
      summary = tf.summary.scalar('learning_rate', lr, epoch)
    super().on_epoch_end(epoch, logs)

  def on_train_end(self, logs=None):
    super().on_train_end(logs)
    self.lr_writer.close()
    


def createLabels(data, classes):
    labels = []
    for filepath in data:
        group = filepath.split(os.path.sep)[-2]
        if group in CLASSES:
            class_num = CLASSES.index(group)
            labels.append(class_num)
    return labels

def normalize(x):
    x = sklearn.preprocessing.minmax_scale(x, feature_range=NORMRANGE)
    return x

def load_and_select_window(filepath, y):
    tf.print("ok")
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


def load_and_select_window_with_scalogram(filepath, y):
    print("Function load_and_select_window_with_scalogram called")  # Debug print
    filepath = tf.compat.as_str_any(filepath)
    pathToPoints = filepath.removesuffix('.npz') + '.txt'
    onsetList = np.loadtxt(pathToPoints).astype(np.int64)
    start_timestep = random.choice(onsetList)
    signal_data = np.load(filepath)['arr_0'].astype('float64')
    
    if NORMALIZE:
        signal_data = normalize(signal_data)
    
    if USE_WINDOWS:
        tf.print(f'{USE_WINDOWS}')
        while (signal_data[start_timestep:]).size < WINDOW_LENGTH:
            start_timestep = random.choice(onsetList)
        signal_data = signal_data[start_timestep:start_timestep + WINDOW_LENGTH]
    
    y = to_categorical(y, num_classes=len(CLASSES))
    
    # Compute scalogram
    Wx, scales, _ = sp.cwt(signal_data, 'morlet', scales='log', derivative=False, nv=32)
    Wx = np.abs(Wx)
    # Convert to image
    Wx_with_channel = Wx[:, :, np.newaxis]
    Wx_rgb = np.concatenate([Wx_with_channel]*3, axis=-1)
    tf.print(f"{Wx_rgb.shape}_conv")
    # Converti l'array in un'immagine PIL
    image_rgb = tf.keras.utils.array_to_img(Wx_rgb)
    image_array = np.array(image_rgb, dtype=np.float32)  # Convert to float32
    image_tensor = tf.convert_to_tensor(np.array(image_rgb), dtype=tf.float32)
    return image_tensor, y

def fix_shape(x, y):
    print(f"Fix_shape called with: {x.shape}, {y.shape}")  # Debug print
    if USE_SCALOGRAM:
        x.set_shape([None, x.shape[0], x.shape[1], x.shape[2]])
        y.set_shape([None, 2])
    
    if USE_WINDOWS and not USE_SCALOGRAM:
        length = WINDOW_LENGTH
        x.set_shape([None, length, 1])
        y.set_shape([None, 2])
    return x, y

def get_id(data_path):
    sub_ids = []
    for item in data_path:
        file_name = os.path.basename(item)  # Ottiene il nome del file dalla directory
        sub_id = file_name[:7]  # Estrae i primi 7 caratteri del nome del file
        if not sub_id.startswith('p'):
            raise Exception(f'Subject name has to start with "p", for example pleth0. Found {sub_id}')
        sub_ids.append(sub_id)
    return sub_ids

def create_train_val_splits(train_paths):
    train_labels = createLabels(train_paths, CLASSES)
    #split train_paths in training and validation sets for k-fold crossvalidation
    groups = get_id(train_paths)
    gkf = GroupKFold(n_splits=K)
    # split is made according subject_ids (group list)
    gkf.get_n_splits(train_paths, train_labels, groups)
    X_train_splits = []
    y_train_splits = []
    X_val_splits = []
    y_val_splits = []
    
    for train_index, val_index in gkf.split(train_paths, train_labels, groups):
        X_train, y_train = np.asarray(train_paths)[train_index], np.asarray(train_labels)[train_index]
        X_val, y_val = np.asarray(train_paths)[val_index], np.asarray(train_labels)[val_index]
        X_train_splits.append(X_train)
        y_train_splits.append(y_train)
        X_val_splits.append(X_val)
        y_val_splits.append(y_val)
    return X_train_splits, y_train_splits, X_val_splits, y_val_splits

def create_dataset(X_train, y_train, X_val, y_val):
    ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    ds_valid = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    tf.print(f'{USE_SCALOGRAM}')
    
    
    if USE_WINDOWS and not USE_SCALOGRAM:
        # Mappatura delle funzioni per il caricamento, il calcolo dello scalogramma e la conversione in immagine
        ds_train = ds_train.map(lambda filepath, label: tf.numpy_function(
            load_and_select_window, [filepath, label], [tf.double, tf.float32]))
    
    if USE_SCALOGRAM:
        tf.print('siamo dentro create datsaset: use scalogram per il ds_train')
        # Mappatura delle funzioni per il caricamento, il calcolo dello scalogramma e la conversione in immagine
        ds_train = ds_train.map(lambda filepath, label: tf.numpy_function(
            load_and_select_window_with_scalogram, [filepath, label], [tf.float32, tf.float32]))

    ds_train = ds_train.cache()
    ds_train = ds_train.batch(BATCH_SIZE)
    ds_train = ds_train.map(fix_shape)

    if USE_WINDOWS and not USE_SCALOGRAM:
        ds_valid = ds_valid.map(lambda filepath, label: tf.numpy_function(
            load_and_select_window, [filepath, label], [tf.double, tf.int32]), 
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    if USE_SCALOGRAM:
        tf.print('siamo dentro create datsaset: use scalogram per il ds_val')
        ds_valid = ds_valid.map(lambda filepath, label: tf.numpy_function(
            load_and_select_window_with_scalogram, [filepath, label], [tf.float32, tf.int32]))

    ds_valid = ds_valid.cache()
    ds_valid = ds_valid.batch(BATCH_SIZE)
    ds_valid = ds_valid.map(fix_shape)
    return ds_train, ds_valid

# create train-validation splits for k-fold cross validation
dataPaths = glob(os.path.join(f'{DATASETDIR}', 'control', '*.npz')) + glob(os.path.join(f'{DATASETDIR}', 'microcirculation', '*.npz'))

print(len(dataPaths))
groups = get_id(dataPaths)
groups = list(np.unique(groups))
print(dataPaths)
print(f"Number of data paths: {len(dataPaths)}")
X_train_splits, y_train_splits, X_val_splits, y_val_splits = create_train_val_splits(dataPaths)

accuracies = []
losses = []
print(f'Running {K}-Fold Cross Validation')

# SELECT SPLIT 
for i in range(0, K):
    print(f'start training on split {i}...')
    # select split
    X_train, y_train = X_train_splits[i], y_train_splits[i]
    X_val , y_val = X_val_splits[i], y_val_splits[i]
    # count sepsis labels in training and validation sets
    class1_train = np.sum(y_train) #conta quanti segnali patologici ci sono per vedere se sono bilanciate le classi
    class1_val = np.sum(y_val)
    print(f'Training class1 percentage is: {100*(class1_train/len(y_train))} %')
    print(f'Training class0 percentage is: {100*((len(y_train) - class1_train)/len(y_train))} %')
    print(f'Validation class1 percentage is: {100*(class1_val/len(y_val))} %')
    print(f'Validation class0 percentage is: {100*((len(y_val) - class1_val)/len(y_val))} %')

    # CREATE DATA SET
    ds_train, ds_valid = create_dataset(X_train, y_train, X_val, y_val)

    if USE_SCALOGRAM:
        # Set the input shape for scalograms
        example_scalogram, _ = next(iter(ds_train))
        INPUT_SHAPE = example_scalogram.shape[1:]
        print(f'{INPUT_SHAPE}')

    elif USE_WINDOWS and not USE_SCALOGRAM:
        INPUT_SHAPE = (WINDOW_LENGTH, 1)
    else:
        INPUT_SHAPE = (NSAMPLES, 1)
    tf.print(f'{INPUT_SHAPE}')
    # CREATE AND COMPILE MODEL
    # To be continued...
    
keras.applications.ResNet50(
include_top=True,
weights="imagenet",
input_tensor=None,
input_shape=None,
pooling=None,
classes=1000,
classifier_activation="softmax",
)
    

