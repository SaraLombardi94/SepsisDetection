# -*- coding: utf-8 -*-


"""
Created on Tue Aug 30 08:53:12 2022

@author: saralombardi
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
from tensorflow.keras.layers import Input, Add, Dense, BatchNormalization, Activation, Dropout, Flatten
from tensorflow.keras.layers import Convolution1D, ZeroPadding1D, MaxPooling1D, AveragePooling1D
from tensorflow.keras.layers import LSTM, TimeDistributed
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
import matplotlib.pyplot as plt

#constants
FS = 125
N_CLASSES = 2 # control, sepsis
LR = 1e-4
BATCH_SIZE = 1
EPOCHS = 100
K = 2
NSAMPLES = FS*30
WINDOW_LENGTH = FS * 30 * 1 
NORMRANGE = (-1,1)
NORMALIZE = False
USE_SHUFFLE = True
USE_WINDOWS = True
USE_JITTER = False
USE_LOSO = False
DROPOUT_RATE = 0.2
RANDOM_STATE = 12
BUFFER_SHUFFLING_SIZE = 180
KERNEL_INITIALIZER='glorot_uniform'
LOSSFUNCTION = tf.keras.losses.BinaryCrossentropy()
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LR)
MODELNAME = f'{K}fold_dsTM2min30_saraCnn1_mse_bs{BATCH_SIZE}_lre{LR}_windows{WINDOW_LENGTH}onset_jitter{USE_JITTER}_ep{EPOCHS}_1'

MODELDIR = r'C:\Users\Utente\Desktop\nuovi segnali\healthy-nonseptic-sepsis\healthy-nonseptic-sepsis\modelli'
DATASETDIR = r'C:\Users\Utente\Desktop\nuovi segnali\healthy-nonseptic-sepsis\healthy-nonseptic-sepsis'
LOGDIR = os.path.join(MODELDIR,MODELNAME,'logs')
WEIGHTSDIR = os.path.join(MODELDIR,MODELNAME,'weights')
CLASSES = ['control','sepsis']
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
    group = filepath.split(os.path.sep)[-3]
    if group in CLASSES:
      class_num = CLASSES.index(group)
      labels.append(class_num)
  return labels

def normalize(x):
   x = sklearn.preprocessing.minmax_scale(x, feature_range = NORMRANGE)
   return x

###  AGGIUSTATA  #####
def load_and_select_window_original(filepath, y):
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

def load_and_select_window(filepath, y):


    # Assicurati che filepath sia una stringa
    filepath = tf.compat.as_str_any(filepath)
    # Costruisci il percorso per i punti di inizio basato sul percorso del file originale
    pathToPoints = filepath.removesuffix('.npz') + '.txt'
    # Carica l'elenco dei punti di inizio
    onsetList = np.loadtxt(pathToPoints).astype(np.int64)
    # Filtra i punti di inizio che permettono una finestra completa
    signal_data = np.load(filepath)['arr_0'].astype('float64')
    max_index = signal_data.shape[0] - WINDOW_LENGTH
    valid_onsetList = onsetList[onsetList <= max_index]

    # Scegli un punto di inizio casuale dai punti validi
    start_timestep = random.choice(valid_onsetList)
    signal_data = np.reshape(signal_data, [signal_data.size, 1])
    if NORMALIZE:
        signal_data = normalize(signal_data)
    if USE_WINDOWS:
        signal_data = signal_data[start_timestep:start_timestep + WINDOW_LENGTH]
    y = to_categorical(y, num_classes=len(CLASSES))
    return signal_data, y

######################

def fix_shape(x, y):
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



def create_in_memory_datasets(X_train_paths, y_train_labels, X_val_paths, y_val_labels):
    # Carica e trasforma i dati di training
    X_train = [load_and_select_window(path,y_train_labels) for path in X_train_paths]
    X_train_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)
    y_train_tensor = tf.convert_to_tensor(y_train_labels, dtype=tf.float32)
    
    # Carica e trasforma i dati di validazione
    X_val = [load_and_select_window(path,y_val_labels) for path in X_val_paths]
    X_val_tensor = tf.convert_to_tensor(X_val, dtype=tf.float32)
    y_val_tensor = tf.convert_to_tensor(y_val_labels, dtype=tf.float32)
    
    # Crea il dataset di training
    ds_train = tf.data.Dataset.from_tensor_slices((X_train_tensor, y_train_tensor))
    ds_train = ds_train.shuffle(buffer_size=len(X_train_paths)).batch(BATCH_SIZE)
    
    # Crea il dataset di validazione
    ds_val = tf.data.Dataset.from_tensor_slices((X_val_tensor, y_val_tensor))
    ds_val = ds_val.batch(BATCH_SIZE)  # Solitamente non si mescola il dataset di validazione
    
    return ds_train, ds_val





def create_dataset(X_train, y_train, X_val, y_val):

  ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
  ds_valid = tf.data.Dataset.from_tensor_slices((X_val, y_val))
  
  # Train dataset
  ds_train = ds_train.map(lambda filepath, label : tf.numpy_function(
        load_and_select_window, [filepath, label], (tf.double, tf.float32)))
  ds_train = ds_train.cache()
  ds_train = ds_train.batch(BATCH_SIZE)
  ds_train = ds_train.map(fix_shape)

  # Validation dataset
  ds_valid = ds_valid.map(lambda filepath, label: tf.numpy_function(
        load_and_select_window, [filepath, label], (tf.double, tf.float32)))
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

# create train-validation splits for k-fold cross validation
dataPaths = glob(os.path.join(f'{DATASETDIR}','sepsis','seed4','*.npz'))+ glob(os.path.join(f'{DATASETDIR}','control','seed4','*.npz'))

print(len(dataPaths))
groups = get_id(dataPaths)
groups = list(np.unique(groups))
print(dataPaths)
print(f"Number of data paths: {len(dataPaths)}")
X_train_splits, y_train_splits, X_val_splits, y_val_splits = create_train_val_splits(dataPaths)
print(len(dataPaths))
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
  
  
  if USE_WINDOWS:
    INPUT_SHAPE = (WINDOW_LENGTH,1)
  else:
    INPUT_SHAPE = (NSAMPLES,1)
  # CREATE AND COMPILE MODEL
  model = ultraSimplifiedSaraCnn1(input_shape = INPUT_SHAPE, nclasses = len(CLASSES))
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
