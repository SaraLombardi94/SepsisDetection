# -*- coding: utf-8 -*-
"""
Created on Tue May 14 15:11:07 2024

@author: Utente
"""

import random
import os
import numpy as np
import tensorflow.keras
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
from tensorflow.keras.layers import Input, Add, Dense, BatchNormalization, Activation, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.layers import Convolution1D, ZeroPadding1D, MaxPooling1D, AveragePooling1D
from tensorflow.keras.layers import LSTM, TimeDistributed
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.initializers import GlorotUniform
import ssqueezepy as sp
from scipy.ndimage import zoom
#constants
FS = 125
N_CLASSES = 2 # control, sepsis 
LR = 1e-3
BATCH_SIZE = 8   #aggiornamento dei pesi della rete 
EPOCHS = 150
K = 5
NSAMPLES = FS*30
WINDOW_LENGTH = FS * 30 * 1 
NORMRANGE = (-1,1)
NORMALIZE = True
USE_SHUFFLE = True
USE_WINDOWS = True
USE_JITTER = False
USE_LOSO = False
USE_SCALOGRAM = True
RESIZE_IMG = False
DROPOUT_RATE = 0.2
RANDOM_STATE = 12
BUFFER_SHUFFLING_SIZE = 180
KERNEL_INITIALIZER='glorot_uniform'
LOSSFUNCTION = tf.keras.losses.BinaryCrossentropy()
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LR)
MODELNAME = f'{K}fold_dsTM2min30_Prova_scalogrammi_MobileNet_3_mse_bs{BATCH_SIZE}_lre{LR}_windows{WINDOW_LENGTH}onset_jitter{USE_JITTER}_ep{EPOCHS}_1'

MODELDIR = r'C:\Users\Utente\Desktop\wetransfer_controls-microcirculation_2024-04-23_1250\controls-microcirculation\tf_bilanciato\modelli'
DATASETDIR = r'C:\Users\Utente\Desktop\wetransfer_controls-microcirculation_2024-04-23_1250\controls-microcirculation\tf_bilanciato'
LOGDIR = os.path.join(MODELDIR,MODELNAME,'logs')
WEIGHTSDIR = os.path.join(MODELDIR,MODELNAME,'weights')
CLASSES = ['control','microcirculation']
#PRETRAINED_MODELPATH = '/Users/saralombardi/Desktop/COVID/pre-trainedsepsi'

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
    
def createLabels(data,classes):
  labels = []
  for filepath in data:
    group = filepath.split(os.path.sep)[-2]
    if group in CLASSES:
      class_num = CLASSES.index(group)
      labels.append(class_num)
  return labels

def normalize(x):
   x = sklearn.preprocessing.minmax_scale(x, feature_range = NORMRANGE)
   return x

###  AGGIUSTATA  #####
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

def resize_array(array, target_height, target_width):
    # Calculate the zoom factors for height and width
    zoom_factors = (target_height / array.shape[0], target_width / array.shape[1], 1)
    
    # Use scipy.ndimage.zoom to resize the array
    resized_array = zoom(array, zoom_factors, order=1)  # Use order=1 for bilinear interpolation
    
    return resized_array

def load_and_select_window_with_scalogram(filepath, y):
    #print("Function load_and_select_window_with_scalogram called")  # Debug print
    filepath = tf.compat.as_str_any(filepath)
    pathToPoints = filepath.removesuffix('.npz') + '.txt'
    onsetList = np.loadtxt(pathToPoints).astype(np.int64)
    start_timestep = random.choice(onsetList)
    signal_data = np.load(filepath)['arr_0'].astype('float64')
    
    if NORMALIZE:
        signal_data = normalize(signal_data)
    
    if USE_WINDOWS:
        #tf.print(f'{USE_WINDOWS}')
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
    Wx_rgb = Wx_rgb.astype(np.float32)
    #tf.print(f"{Wx_rgb.shape}_conv")

    if RESIZE_IMG:
        resized_image = resize_array(Wx_rgb, 224, 224)
        return resized_image, y
    else:    
        return Wx_rgb, y


######################

def fix_shape(x, y):
  print(f"Fix_shape called with: {x.shape}, {y.shape}")
  if USE_SCALOGRAM:
      x.set_shape([None, x.shape[0], x.shape[1], x.shape[2]])
      y.set_shape([None, 2])
  
  if USE_WINDOWS and not USE_SCALOGRAM:
    length = WINDOW_LENGTH
    x.set_shape([None, length, 1])
    y.set_shape([None, 2])

  return x, y


# augmenting functions 
def calculate_fft(x, y):
    xf = fft(x)
    return xf, y
    
def amplitudeScaling(x, y):
    choice = np.random.rand()
    if choice < 0.5:
       return x, y
    else:
        sigma=0.01
        factor = np.random.normal(loc=1., scale=sigma, size=x.shape)
        scaled = np.multiply(x, factor)
    return scaled, y

def jitter(x, y):
  # Add random 0.5 possibility of applying the augmentation
  choice = np.random.rand()
  if choice < 0.5:
     return x, y
  else:
    sigma = 0.01
    x_jitter = x + np.random.normal(loc=0., scale=sigma, size=x.shape)
    return x_jitter, y


####### AGGIUSTATA ##########
def get_id(data_path):
    sub_ids = []
    for item in data_path:
        file_name = os.path.basename(item)  # Ottiene il nome del file dalla directory
        sub_id = file_name[:7]  # Estrae i primi 7 caratteri del nome del file
        if not sub_id.startswith('p'):
            raise Exception(f'Subject name has to start with "p", for example pleth0. Found {sub_id}')
        sub_ids.append(sub_id)
    return sub_ids


# Funzione per plottare le immagini scalogrammi nel dataset
def plot_dataset_samples(dataset, num_samples=5):
    for i, (image_batch, label_batch) in enumerate(dataset.take(num_samples)):
        for j in range(image_batch.shape[0]):  # Itera attraverso ogni immagine nel batch
            image = image_batch[j].numpy()
            label = label_batch[j].numpy()
            plt.figure(figsize=(8, 4))
            plt.imshow(image.astype('float32'), aspect='auto', cmap='viridis')
            plt.title(f'Sample {i+1}-{j+1} - Label: {CLASSES[np.argmax(label)]}')
            plt.colorbar()
            plt.show()
        
        

###################

##### DOVREBBE ANDARE BENE #####
def create_train_val_splits(train_paths):
  train_labels = createLabels(train_paths,CLASSES)
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
      #X_train, y_train = shuffle(X_train, y_train, random_state=RANDOM_STATE)
      X_val, y_val = np.asarray(train_paths)[val_index], np.asarray(train_labels)[val_index]
      #X_val, y_val = shuffle(X_val, y_val, random_state=RANDOM_STATE)
      X_train_splits.append(X_train)
      y_train_splits.append(y_train)
      X_val_splits.append(X_val)
      y_val_splits.append(y_val)
  return X_train_splits, y_train_splits, X_val_splits, y_val_splits



def create_dataset(X_train, y_train, X_val, y_val):

  ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
  ds_valid = tf.data.Dataset.from_tensor_slices((X_val, y_val))
  
  if USE_WINDOWS and not USE_SCALOGRAM:
      # Mappatura delle funzioni per il caricamento, il calcolo dello scalogramma e la conversione in immagine
      ds_train = ds_train.map(lambda filepath, label: tf.numpy_function(
            load_and_select_window, [filepath, label], [tf.double, tf.float32]))
  if USE_SCALOGRAM:
      # Mappatura delle funzioni per il caricamento, il calcolo dello scalogramma e la conversione in immagine
      ds_train = ds_train.map(lambda filepath, label: tf.numpy_function(
            load_and_select_window_with_scalogram, [filepath, label], [tf.float32, tf.float32]))

  ds_train = ds_train.cache()

  ds_train = ds_train.batch(BATCH_SIZE)

  ds_train = ds_train.map(fix_shape)


  if USE_WINDOWS and not USE_SCALOGRAM:
        ds_valid = ds_valid.map(lambda filepath, label: tf.numpy_function(
            load_and_select_window, [filepath, label], [tf.double, tf.float32]), 
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
  if USE_SCALOGRAM:
        ds_valid = ds_valid.map(lambda filepath, label: tf.numpy_function(
            load_and_select_window_with_scalogram, [filepath, label], [tf.float32, tf.float32]))


  
  ds_valid = ds_valid.cache()
  ds_valid = ds_valid.batch(BATCH_SIZE)
  ds_valid = ds_valid.map(fix_shape)
  return ds_train, ds_valid


# DEFINE MODELS

############    LA PRIMA DA PROVARE      #################

def saraCnn1(input_shape, nclasses):
    x_input = Input(input_shape)    
    x = Convolution1D(filters=64, kernel_size=11, strides=1, activation='relu', kernel_initializer=KERNEL_INITIALIZER) (x_input)
    x = MaxPooling1D(pool_size=(4),padding='valid',strides=2)(x)
    x = Convolution1D(filters=64, kernel_size=11, strides=1, activation='relu', kernel_initializer=KERNEL_INITIALIZER) (x)
    x = MaxPooling1D(pool_size=(4),padding='valid',strides=2)(x)
    x = Convolution1D(filters=128, kernel_size=11, strides=1, activation='relu', kernel_initializer=KERNEL_INITIALIZER) (x)
    x = MaxPooling1D(pool_size=(4),padding='valid',strides=2)(x)
    x = Convolution1D(filters=128, kernel_size=11, strides=1, activation='relu', kernel_initializer=KERNEL_INITIALIZER) (x)
    x = MaxPooling1D(pool_size=(4),padding='valid',strides=2)(x)
    x = Flatten()(x)
    x = Dense(100, activation = 'relu', kernel_initializer=KERNEL_INITIALIZER)(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = Dense(50, activation = 'relu', kernel_initializer=KERNEL_INITIALIZER)(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = Dense(nclasses, activation = 'softmax', kernel_initializer=KERNEL_INITIALIZER)(x)
    #create model
    model = Model(inputs = x_input, outputs = x , name = 'eegseizure1')
    return model


def saraCnnEnhanced(input_shape, nclasses):
    x_input = Input(input_shape)
    
    # Primo blocco convoluzionale con Batch Normalization e MaxPooling
    x = Convolution1D(filters=64, kernel_size=11, strides=1, padding='same', kernel_initializer='he_uniform')(x_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=4, strides=2, padding='valid')(x)
    
    # Secondo blocco convoluzionale con Batch Normalization e AveragePooling
    x = Convolution1D(filters=128, kernel_size=7, strides=1, padding='same', kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling1D(pool_size=4, strides=2, padding='valid')(x)
    
    # Terzo blocco convoluzionale con Batch Normalization e MaxPooling
    x = Convolution1D(filters=256, kernel_size=5, strides=1, padding='same', kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=4, strides=2, padding='valid')(x)
    
    # Quarto blocco convoluzionale con Batch Normalization e AveragePooling
    x = Convolution1D(filters=256, kernel_size=5, strides=1, padding='same', kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling1D(pool_size=4, strides=2, padding='valid')(x)

    # Appiattimento e strati fully connected con Dropout
    x = Flatten()(x)
    x = Dense(256, activation='relu', kernel_initializer='he_uniform')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu', kernel_initializer='he_uniform')(x)
    x = Dropout(0.5)(x)
    
    # Strato di output
    x = Dense(nclasses, activation='softmax', kernel_initializer='he_uniform')(x)
    
    # Creazione del modello
    model = Model(inputs=x_input, outputs=x, name='Enhanced_saraCnn')
    return model

def simplifiedSaraCnn1(input_shape, nclasses, KERNEL_INITIALIZER='glorot_uniform', DROPOUT_RATE=0.5):
    x_input = Input(input_shape)    
    x = Convolution1D(filters=32, kernel_size=11, strides=2, activation='relu', kernel_initializer=KERNEL_INITIALIZER)(x_input)
    x = MaxPooling1D(pool_size=4, padding='valid', strides=2)(x)
    x = Convolution1D(filters=64, kernel_size=11, strides=2, activation='relu', kernel_initializer=KERNEL_INITIALIZER)(x)
    x = MaxPooling1D(pool_size=4, padding='valid', strides=2)(x)
    x = Flatten()(x)
    x = Dense(50, activation='relu', kernel_initializer=KERNEL_INITIALIZER)(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = Dense(nclasses, activation='softmax', kernel_initializer=KERNEL_INITIALIZER)(x)
    model = Model(inputs=x_input, outputs=x, name='simplified_eegseizure1')
    return model

def ultraSimplifiedSaraCnn1(input_shape, nclasses, KERNEL_INITIALIZER='glorot_uniform', DROPOUT_RATE=0.5):
    x_input = Input(input_shape)    
    x = Convolution1D(filters=16, kernel_size=11, strides=2, activation='relu', kernel_initializer=KERNEL_INITIALIZER)(x_input)
    x = MaxPooling1D(pool_size=4, padding='valid', strides=2)(x)
    x = Flatten()(x)
    x = Dense(nclasses, activation='softmax', kernel_initializer=KERNEL_INITIALIZER)(x)
    model = Model(inputs=x_input, outputs=x, name='ultraSimplified_eegseizure1')
    return model



def saraRNN1(input_shape, nclasses):
    x_input = Input(input_shape)

    # Definizione degli strati LSTM
    x = LSTM(64, return_sequences=True, kernel_initializer=GlorotUniform())(x_input)
    x = Dropout(0.5)(x)
    x = LSTM(64, return_sequences=True, kernel_initializer=GlorotUniform())(x)
    x = Dropout(0.5)(x)
    x = LSTM(128, return_sequences=True, kernel_initializer=GlorotUniform())(x)
    x = Dropout(0.5)(x)
    x = LSTM(128, return_sequences=False, kernel_initializer=GlorotUniform())(x)  # return_sequences=False per preparare l'uscita all'ultimo layer Dense

    # Aggiunta dei livelli fully connected
    x = Dense(100, activation='relu', kernel_initializer=GlorotUniform())(x)
    x = Dropout(0.5)(x)
    x = Dense(50, activation='relu', kernel_initializer=GlorotUniform())(x)
    x = Dropout(0.5)(x)

    # Strato di output
    x = Dense(nclasses, activation='softmax', kernel_initializer=GlorotUniform())(x)

    # Creazione del modello
    model = Model(inputs=x_input, outputs=x, name='eegseizure_rnn1')
    return model

def build_brnn_model(input_shape, nclasses):
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(nclasses, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
 

def gruSara(input_shape,nclasses):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(16, 32, activation='relu', input_shape=input_shape, kernel_regularizer='l2'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=4))
    #model.add(tf.keras.layers.BatchNormalization(axis=1))
    model.add(tf.keras.layers.Conv1D(32, 16, activation='relu', kernel_regularizer='l2'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=4))
    #model.add(tf.keras.layers.BatchNormalization(axis=1))
    model.add(tf.keras.layers.Conv1D(64, 8, activation='relu', kernel_regularizer='l2'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model.add(tf.keras.layers.BatchNormalization(axis=1))
    model.add(tf.keras.layers.GRU(16, dropout=0.5))
    model.add(tf.keras.layers.Dense(nclasses, activation='sigmoid'))
    return model

def gruSara_plus(input_shape, nclasses):
    model = tf.keras.Sequential()
    # Layer convoluzionali iniziali
    model.add(tf.keras.layers.Conv1D(16, 32, activation='relu', input_shape=input_shape, kernel_regularizer='l2'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=4))
    model.add(tf.keras.layers.Conv1D(32, 16, activation='relu', kernel_regularizer='l2'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=4))
    model.add(tf.keras.layers.Conv1D(64, 8, activation='relu', kernel_regularizer='l2'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model.add(tf.keras.layers.BatchNormalization())

    # Aggiunta di ulteriori layer convoluzionali
    model.add(tf.keras.layers.Conv1D(128, 4, activation='relu', kernel_regularizer='l2'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=2))

    # GRU e layer densamente connessi
    model.add(tf.keras.layers.GRU(16, dropout=0.5, return_sequences=False))  # 'return_sequences=False' per prepararsi al Dense
    model.add(tf.keras.layers.Dense(nclasses, activation='sigmoid'))

    return model

def gruSara_plus_adjusted(input_shape, nclasses):
    model = tf.keras.Sequential()
    # Layer convoluzionali iniziali
    model.add(tf.keras.layers.Conv1D(16, 32, activation='relu', input_shape=input_shape, kernel_regularizer='l2'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=4))
    model.add(tf.keras.layers.Conv1D(32, 16, activation='relu', kernel_regularizer='l2'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=4))
    model.add(tf.keras.layers.Conv1D(64, 8, activation='relu', kernel_regularizer='l2'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model.add(tf.keras.layers.BatchNormalization())

    # Nuovo layer convoluzionale aggiunto
    model.add(tf.keras.layers.Conv1D(64, 16, activation='relu', kernel_regularizer='l2'))  # Filtro più grande
    model.add(tf.keras.layers.MaxPool1D(pool_size=2))

    # Aggiunta di ulteriori layer convoluzionali esistenti
    model.add(tf.keras.layers.Conv1D(128, 4, activation='relu', kernel_regularizer='l2'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=2))

    # Nuovo layer GRU con più unità
    model.add(tf.keras.layers.GRU(32, dropout=0.5, return_sequences=False))  # Più unità GRU

    # Layer densamente connesso finale
    model.add(tf.keras.layers.Dense(nclasses, activation='sigmoid'))

    return model


def gruSara_plus_adjusted_64(input_shape, nclasses):
    model = tf.keras.Sequential()
    # Layer convoluzionali iniziali
    model.add(tf.keras.layers.Conv1D(16, 32, activation='relu', input_shape=input_shape, kernel_regularizer='l2'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=4))
    model.add(tf.keras.layers.Conv1D(32, 16, activation='relu', kernel_regularizer='l2'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=4))
    model.add(tf.keras.layers.Conv1D(64, 8, activation='relu', kernel_regularizer='l2'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model.add(tf.keras.layers.BatchNormalization())

    # Nuovo layer convoluzionale aggiunto
    model.add(tf.keras.layers.Conv1D(64, 16, activation='relu', kernel_regularizer='l2'))  # Filtro più grande
    model.add(tf.keras.layers.MaxPool1D(pool_size=2))

    # Aggiunta di ulteriori layer convoluzionali esistenti
    model.add(tf.keras.layers.Conv1D(128, 4, activation='relu', kernel_regularizer='l2'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=2))

    # Nuovo layer GRU con più unità
    model.add(tf.keras.layers.GRU(64, dropout=0.5, return_sequences=False))  # Più unità GRU

    # Layer densamente connesso finale
    model.add(tf.keras.layers.Dense(nclasses, activation='sigmoid'))

    return model


def modello_scalogrammi_ResNet50(input_shape, nclasses):
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling= None
    )
    base_model.trainable = False  # Freeze the base model

    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(DROPOUT_RATE)(x)
    outputs = Dense(nclasses, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=OPTIMIZER, loss = LOSSFUNCTION, metrics=['accuracy'])
    return model



def modello_scalogrammi_MobileNetV2(input_shape, nclasses, unfrozen_layers=50):
    base_model = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling=None
    )

    # Freeze all layers first
    base_model.trainable = False

    # Unfreeze the last 'unfrozen_layers' layers
    for layer in base_model.layers[-unfrozen_layers:]:
        layer.trainable = True

    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=True)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(DROPOUT_RATE)(x)  # Assuming DROPOUT_RATE is 0.5
    outputs = Dense(nclasses, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # Assuming OPTIMIZER is 'adam' and LOSSFUNCTION is 'sparse_categorical_crossentropy'
    return model


# create train-validation splits for k-fold cross validation
dataPaths = glob(os.path.join(f'{DATASETDIR}','control','*.npz'))+ glob(os.path.join(f'{DATASETDIR}','microcirculation','*.npz'))

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
  
  # if USE_SCALOGRAM:
  #   # Per plottare le prime N immagini dal dataset di training
  #   plot_dataset_samples(ds_train, num_samples=5)
    
    
  if USE_SCALOGRAM:
        # Set the input shape for scalograms
        example_scalogram, _ = next(iter(ds_train))
        #print(f'{example_scalogram}')
        INPUT_SHAPE = example_scalogram.shape[1:]
        #print(f'{INPUT_SHAPE}')

            
  
  
  
  if USE_WINDOWS and not USE_SCALOGRAM:
    INPUT_SHAPE = (WINDOW_LENGTH,1)
    
  if not USE_WINDOWS and not USE_SCALOGRAM:
    INPUT_SHAPE = (NSAMPLES,1)
    
    
    
  # CREATE AND COMPILE MODEL
  model = modello_scalogrammi_MobileNetV2(input_shape = INPUT_SHAPE, nclasses = len(CLASSES))
  model.summary()
  model.compile(optimizer=OPTIMIZER,loss=LOSSFUNCTION, metrics=['accuracy'])


  # CREATE CALLBACKS
  tensorboard = LRTensorBoard(log_dir=os.path.join(LOGDIR,f'{i}fold'),histogram_freq=0)

  model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(WEIGHTSDIR, f'{i}fold.keras'),
    save_weights_only =False,
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    verbose = 1
  )
  reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.05,
                                patience=10, min_lr=1e-7, verbose = 1)

  def scheduleLR (epoch, lr):
    if epoch <50:
      return lr
    elif epoch > 50 and epoch < 600:
      return np.float64(1e-6)
    else:
      return np.float64(1e-6)

  lrScheduler = LearningRateScheduler(
      scheduleLR, verbose=1
  )

  callbacks = [tensorboard, model_checkpoint]

  #### TRAINING ####
  print('start training...')
  history = model.fit(ds_train,
            epochs = EPOCHS,
            verbose = 1,
            validation_data = ds_valid,
            callbacks=[tensorboard, model_checkpoint])
  
  np.savetxt(os.path.join(LOGDIR,f'{i}fold',f'X_TRAIN{i}.txt'), X_train, fmt='%s')
  np.savetxt(os.path.join(LOGDIR,f'{i}fold',f'X_val{i}.txt'), X_val, fmt='%s')
  model.save(os.path.join(MODELDIR,MODELNAME,f'{i}fold.h5'))

  #### TESTING ####
  print('start testing...')
  modelB = tf.keras.models.load_model(os.path.join(WEIGHTSDIR,f'{i}fold.keras'))
  score = modelB.evaluate(ds_valid, verbose=1, callbacks=[tensorboard])
  print('Val loss:', score[0])
  print('Val accuracy:', score[1])
  accuracies.append(score[1])
  losses.append(score[0])

np.savetxt(os.path.join(MODELDIR,MODELNAME,'accuracies.txt'), accuracies)
np.savetxt(os.path.join(MODELDIR,MODELNAME,'losses.txt'), losses)
print(f'Mean accuracy is : {np.mean(accuracies)}')
print(f'Mean loss is : {np.mean(losses)}')


##########saliency map #########
sample_signal, classe = load_and_select_window('C:/Users/Utente/Desktop/wetransfer_controls-microcirculation_2024-04-23_1250/controls-microcirculation/tf_bilanciato/microcirculation/plre90.npy_#99.npz',1)

# Assumi che 'model' sia il tuo modello già addestrato e 'sample_signal' sia un segnale PPG che hai già preparato e normalizzato
sample_signal = tf.convert_to_tensor(sample_signal, dtype=tf.float32)
sample_signal = tf.expand_dims(sample_signal, axis=0)  # Aggiungi una dimensione batch se necessario

with tf.GradientTape() as tape:
    tape.watch(sample_signal)
    predictions = model(sample_signal)
    class_idx = tf.argmax(predictions[0])  # Ottieni l'indice della classe con la probabilità più alta
    output = predictions[:, class_idx]

# Calcola i gradienti dell'output rispetto all'input
gradients = tape.gradient(output, sample_signal)

# Calcola il valore assoluto dei gradienti e fai la media lungo l'asse del canale se necessario
grad_abs = tf.reduce_mean(tf.abs(gradients), axis=-1)

# Riduci la dimensione dell'array a 2D se necessario
if grad_abs.ndim > 1:
    grad_abs = tf.squeeze(grad_abs)

# Assicurati che grad_abs sia un array 2D aggiungendo una nuova dimensione se è un array 1D
if grad_abs.ndim == 1:
    grad_abs = grad_abs[None, :]  # Aggiungi una dimensione fittizia per renderlo 2D

# Visualizza il segnale e la saliency map
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title('Original PPG Signal')
plt.plot(np.squeeze(sample_signal.numpy()))  # Converti il tensore in un array NumPy per il plotting

plt.subplot(1, 2, 2)
plt.title('Saliency Map')
# Usa imshow per creare un mappable
im = plt.imshow(grad_abs, aspect='auto', cmap='viridis', extent=[0, grad_abs.shape[-1], 0, 1])
plt.colorbar(im)  # Aggiungi la colorbar riferendoti a 'im'
plt.show()