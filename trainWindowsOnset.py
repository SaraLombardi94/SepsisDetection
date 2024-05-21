#!/usr/bin/env python3
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
import sklearn
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold, train_test_split, GroupShuffleSplit, StratifiedGroupKFold, GroupKFold
from sklearn import preprocessing
from sklearn.utils import shuffle
from scipy.fft import fft
#model
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Add, Dense, BatchNormalization, Activation, Dropout, Flatten
from tensorflow.keras.layers import Convolution1D, ZeroPadding1D, MaxPooling1D, AveragePooling1D,Multiply, Permute, Reshape, Lambda, RepeatVector
from tensorflow.keras.layers import LSTM, TimeDistributed
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
import matplotlib.pyplot as plt
import keras.backend as K
import logging
logging.basicConfig(level=logging.INFO)
#constants
FS = 125
N_CLASSES = 2 # control, sepsis
LR = 1e-5
BATCH_SIZE = 32
EPOCHS = 150
K = 5
NSAMPLES = FS*30
WINDOW_LENGTH = FS * 30 * 1 
NORMRANGE = (-1,1)
NORMALIZE = True
USE_SHUFFLE = True
USE_WINDOWS = True
USE_FFT = False
USE_JITTER = False
USE_SCALING = False
USE_LOSO = False
DROPOUT_RATE = 0.3
RANDOM_STATE = 12
BUFFER_SHUFFLING_SIZE = 180
KERNEL_INITIALIZER='glorot_uniform'
LOSSFUNCTION = tf.keras.losses.BinaryCrossentropy()
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LR)
MODELNAME = f'k{K}_gruSara4_lr{LR}_ep{EPOCHS}_'

MODELDIR = r'D:\phD_Sara\models\control_target'
DATASETDIR = r'D:\phD_Sara\microcircolo\Sepsis\datasets\controls-microcirculation\datasetSeed4'
LOGDIR = os.path.join(MODELDIR,MODELNAME,'logs')
WEIGHTSDIR = os.path.join(MODELDIR,MODELNAME,'weights')
CLASSES = ['controls','target']
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

def load_and_select_window(x,y):
    #pathToPoints = x.removesuffix(b'.npz')
    pathToPoints = tf.compat.as_str_any(x)
    # Costruisci il percorso per i punti di inizio basato sul percorso del file originale
    pathToPoints = pathToPoints.removesuffix('.npz') + '.txt'
    onsetList = np.loadtxt(pathToPoints)
    onsetList = onsetList.astype('int')
    start_timestep = random.choice(onsetList)
    x = np.load(x)['arr_0'].astype('float64')
    x = np.reshape(x,[x.size,1])
    if NORMALIZE:
        x = normalize(x)
    if USE_WINDOWS:
        while((x[start_timestep:]).size < WINDOW_LENGTH):
            start_timestep = random.choice(onsetList)
            
        x = x[start_timestep:start_timestep + WINDOW_LENGTH]
    y = to_categorical(y,num_classes=len(CLASSES))
    return x, y

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

def get_id(data_path):
  sub_ids=[]
  for item in data_path:
    file_name = item.split(os.path.sep)[-1].rstrip('.npz')
    sub_id = file_name.split('_')[0]
    if not sub_id.startswith('p'):
      raise Exception(f'Subject name has to start with "p", for example pleth0. Found {sub_id}')
    sub_ids.append(sub_id)
  return sub_ids
   

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
      X_train, y_train = shuffle(X_train, y_train, random_state=RANDOM_STATE)
      X_val, y_val = np.asarray(train_paths)[val_index], np.asarray(train_labels)[val_index]
      X_val, y_val = shuffle(X_val, y_val, random_state=RANDOM_STATE)
      X_train_splits.append(X_train)
      y_train_splits.append(y_train)
      X_val_splits.append(X_val)
      y_val_splits.append(y_val)
  return X_train_splits, y_train_splits, X_val_splits, y_val_splits

#STRATIFIED GROUP K-FOLD
# =============================================================================
# 
# def create_train_val_splits(train_paths):
#   train_labels = createLabels(train_paths,CLASSES)
#   #split train_paths in training and validation sets for k-fold crossvalidation
#   groups = get_id(train_paths)
#   gkf = StratifiedGroupKFold(n_splits=K)
#   # split is made according subject_ids (group list)
#   gkf.get_n_splits(train_paths, train_labels, groups)
#   X_train_splits = []
#   y_train_splits = []
#   X_val_splits = []
#   y_val_splits = []
#  
#   for train_index, val_index in gkf.split(train_paths, train_labels, groups):
#      X_train, y_train = np.asarray(train_paths)[train_index], np.asarray(train_labels)[train_index]
#      X_train, y_train = shuffle(X_train, y_train, random_state=RANDOM_STATE)
#      X_val, y_val = np.asarray(train_paths)[val_index], np.asarray(train_labels)[val_index]
#      X_val, y_val = shuffle(X_val, y_val, random_state=RANDOM_STATE)
#      X_train_splits.append(X_train)
#      y_train_splits.append(y_train)
#      X_val_splits.append(X_val)
#      y_val_splits.append(y_val)
#   return X_train_splits, y_train_splits, X_val_splits, y_val_splits
# =============================================================================


def create_dataset(X_train, y_train, X_val, y_val):

  ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
  ds_valid = tf.data.Dataset.from_tensor_slices((X_val, y_val))
  

  # Train dataset
  ds_train = ds_train.map(lambda filepath, label : tf.numpy_function(
        load_and_select_window, [filepath, label], (tf.double, tf.float32)))
# =============================================================================
#   if NORMALIZE:
#   ds_train = ds_train.map(lambda filepath, label: tf.numpy_function(
#         normalize, [filepath, label], (tf.double, tf.float32)))
# 
# =============================================================================
  
  ds_train = ds_train.cache()
  if USE_SHUFFLE:
    ds_train = ds_train.shuffle(BUFFER_SHUFFLING_SIZE)
  if USE_FFT:
    ds_train = ds_train.map(lambda x, y: tf.numpy_function(
      calculate_fft, [x, y], (tf.complex128, tf.float32)))

  if USE_JITTER:
    ds_train = ds_train.map(lambda x, y: tf.numpy_function(
            jitter, [x, y], (tf.double, tf.float32)))
    ds_train = ds_train.map(lambda x, y: tf.numpy_function(
            normalize, [x, y], (tf.double, tf.float32)))
  if USE_SCALING:
    ds_train = ds_train.map(lambda x, y: tf.numpy_function(
            amplitudeScaling, [x, y], (tf.double, tf.float32)))
    ds_train = ds_train.map(lambda x, y: tf.numpy_function(
            normalize, [x, y], (tf.double, tf.float32)))
  ds_train = ds_train.batch(BATCH_SIZE)
  ds_train = ds_train.map(fix_shape)

  # Validation dataset
  ds_valid = ds_valid.map(lambda filepath, label: tf.numpy_function(
        load_and_select_window, [filepath, label], (tf.double, tf.float32)))
  
# =============================================================================
# 
#   ds_valid = ds_valid.map(lambda filepath, label: tf.numpy_function(
#         normalize, [filepath, label], (tf.double, tf.float32)))
# =============================================================================
  
  ds_valid = ds_valid.cache()

  if USE_FFT:
    ds_valid = ds_valid.map(lambda x, y: tf.numpy_function(
      calculate_fft, [x, y], (tf.complex128, tf.float32)))

  ds_valid = ds_valid.batch(BATCH_SIZE)
  ds_valid = ds_valid.map(fix_shape)
  return ds_train, ds_valid

# DEFINE MODELS

def cnnAgePPG(input_shape, nclasses):
    x_input = Input(input_shape)    
    x = Convolution1D(filters=10, kernel_size=6, strides=1, activation='relu', kernel_initializer=KERNEL_INITIALIZER) (x_input)
    x = Convolution1D(filters=8, kernel_size=5, strides=1, activation='relu', kernel_initializer=KERNEL_INITIALIZER) (x)
    x = Flatten()(x)
    x = Dense(1024, activation = 'relu', kernel_initializer=KERNEL_INITIALIZER)(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = Dense(1024, activation = 'relu', kernel_initializer=KERNEL_INITIALIZER)(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = Dense(nclasses, activation = 'softmax', kernel_initializer=KERNEL_INITIALIZER)(x)
    #create model
    model = Model(inputs = x_input, outputs = x , name = 'eegseizure')
    return model 

def eegSeizureModel(input_shape, nclasses):
    x_input = Input(input_shape)    
    x = Convolution1D(filters=4, kernel_size=6, strides=1, activation='relu', kernel_initializer=KERNEL_INITIALIZER) (x_input)
    x = MaxPooling1D(pool_size=(2),padding='valid',strides=2)(x)
    x = Convolution1D(filters=4, kernel_size=5, strides=1, activation='relu', kernel_initializer=KERNEL_INITIALIZER) (x)
    x = MaxPooling1D(pool_size=(2),padding='valid',strides=2)(x)
    x = Convolution1D(filters=10, kernel_size=4, strides=1, activation='relu', kernel_initializer=KERNEL_INITIALIZER) (x)
    x = MaxPooling1D(pool_size=(2),padding='valid',strides=2)(x)
    x = Convolution1D(filters=10, kernel_size=4, strides=1, activation='relu', kernel_initializer=KERNEL_INITIALIZER) (x)
    x = MaxPooling1D(pool_size=(2),padding='valid',strides=2)(x)
    x = Convolution1D(filters=15, kernel_size=4, strides=1, activation='relu',kernel_initializer=KERNEL_INITIALIZER) (x)
    x = MaxPooling1D(pool_size=(2),padding='valid',strides=2)(x)
    x = Flatten()(x)
    x = Dense(50, activation = 'relu', kernel_initializer=KERNEL_INITIALIZER)(x)
    x = Dense(20, activation = 'relu', kernel_initializer=KERNEL_INITIALIZER)(x)
    x = Dense(nclasses, activation = 'softmax', kernel_initializer=KERNEL_INITIALIZER)(x)
    #create model
    model = Model(inputs = x_input, outputs = x , name = 'eegseizure')
    return model 

def eegSeizureModel1(input_shape, nclasses):
    x_input = Input(input_shape)    
    x = Convolution1D(filters=4, kernel_size=11, strides=1, activation='relu', kernel_initializer=KERNEL_INITIALIZER) (x_input)
    x = MaxPooling1D(pool_size=(2),padding='valid',strides=2)(x)
    x = Convolution1D(filters=4, kernel_size=7, strides=1, activation='relu', kernel_initializer=KERNEL_INITIALIZER) (x)
    x = MaxPooling1D(pool_size=(2),padding='valid',strides=2)(x)
    x = Convolution1D(filters=10, kernel_size=5, strides=1, activation='relu', kernel_initializer=KERNEL_INITIALIZER) (x)
    x = MaxPooling1D(pool_size=(2),padding='valid',strides=2)(x)
    x = Convolution1D(filters=10, kernel_size=3, strides=1, activation='relu', kernel_initializer=KERNEL_INITIALIZER) (x)
    x = MaxPooling1D(pool_size=(2),padding='valid',strides=2)(x)
    x = Convolution1D(filters=15, kernel_size=3, strides=1, activation='relu',kernel_initializer=KERNEL_INITIALIZER) (x)
    x = MaxPooling1D(pool_size=(2),padding='valid',strides=2)(x)
    x = Flatten()(x)
    x = Dense(50, activation = 'relu', kernel_initializer=KERNEL_INITIALIZER)(x)
    x = Dense(20, activation = 'relu', kernel_initializer=KERNEL_INITIALIZER)(x)
    x = Dense(nclasses, activation = 'softmax', kernel_initializer=KERNEL_INITIALIZER)(x)
    #create model
    model = Model(inputs = x_input, outputs = x , name = 'eegseizure1')
    return model

def eegSeizureModel3(input_shape, nclasses):
    x_input = Input(input_shape)    
    x = Convolution1D(filters=8, kernel_size=11, strides=1, activation='relu', kernel_initializer=KERNEL_INITIALIZER) (x_input)
    x = MaxPooling1D(pool_size=(2),padding='valid',strides=2)(x)
    x = Convolution1D(filters=8, kernel_size=7, strides=1, activation='relu', kernel_initializer=KERNEL_INITIALIZER) (x)
    x = MaxPooling1D(pool_size=(2),padding='valid',strides=2)(x)
    x = Convolution1D(filters=16, kernel_size=5, strides=1, activation='relu', kernel_initializer=KERNEL_INITIALIZER) (x)
    x = MaxPooling1D(pool_size=(2),padding='valid',strides=2)(x)
    x = Convolution1D(filters=16, kernel_size=3, strides=1, activation='relu', kernel_initializer=KERNEL_INITIALIZER) (x)
    x = MaxPooling1D(pool_size=(2),padding='valid',strides=2)(x)
    x = Convolution1D(filters=32, kernel_size=3, strides=1, activation='relu',kernel_initializer=KERNEL_INITIALIZER) (x)
    x = MaxPooling1D(pool_size=(2),padding='valid',strides=2)(x)
    x = Convolution1D(filters=32, kernel_size=3, strides=1, activation='relu',kernel_initializer=KERNEL_INITIALIZER) (x)
    x = MaxPooling1D(pool_size=(2),padding='valid',strides=2)(x)
    x = Flatten()(x)
    x = Dense(100, activation = 'relu', kernel_initializer=KERNEL_INITIALIZER)(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = Dense(500, activation = 'relu', kernel_initializer=KERNEL_INITIALIZER)(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = Dense(nclasses, activation = 'softmax', kernel_initializer=KERNEL_INITIALIZER)(x)
    #create model
    model = Model(inputs = x_input, outputs = x , name = 'eegseizure1')
    return model


def eegSeizureModel6(input_shape, nclasses):
    x_input = Input(input_shape)    
    x = Convolution1D(filters=8, kernel_size=11, strides=1, activation='relu', kernel_initializer=KERNEL_INITIALIZER) (x_input)
    x = MaxPooling1D(pool_size=(2),padding='valid',strides=2)(x)
    x = Convolution1D(filters=8, kernel_size=11, strides=1, activation='relu', kernel_initializer=KERNEL_INITIALIZER) (x)
    x = MaxPooling1D(pool_size=(2),padding='valid',strides=2)(x)
    x = Convolution1D(filters=16, kernel_size=11, strides=1, activation='relu', kernel_initializer=KERNEL_INITIALIZER) (x)
    x = MaxPooling1D(pool_size=(2),padding='valid',strides=2)(x)
    x = Convolution1D(filters=16, kernel_size=11, strides=1, activation='relu', kernel_initializer=KERNEL_INITIALIZER) (x)
    x = MaxPooling1D(pool_size=(2),padding='valid',strides=2)(x)
    x = Convolution1D(filters=32, kernel_size=11, strides=1, activation='relu',kernel_initializer=KERNEL_INITIALIZER) (x)
    x = MaxPooling1D(pool_size=(2),padding='valid',strides=2)(x)
    x = Convolution1D(filters=32, kernel_size=11, strides=1, activation='relu',kernel_initializer=KERNEL_INITIALIZER) (x)
    x = MaxPooling1D(pool_size=(2),padding='valid',strides=2)(x)
    x = Flatten()(x)
    x = Dense(100, activation = 'relu', kernel_initializer=KERNEL_INITIALIZER)(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = Dense(50, activation = 'relu', kernel_initializer=KERNEL_INITIALIZER)(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = Dense(nclasses, activation = 'softmax', kernel_initializer=KERNEL_INITIALIZER)(x)
    #create model
    model = Model(inputs = x_input, outputs = x , name = 'eegseizure1')
    return model

def attention_block(inputs):
    # a simple attention mechanism with a single dense layer
    attention_probs = Dense(inputs.shape[-1], activation='softmax', name='attention_vec')(inputs)
    attention_mul = Multiply()([inputs, attention_probs])
    return attention_mul

def saraCnn1(input_shape, nclasses):
    x_input = Input(input_shape)
    x = Convolution1D(filters=64, kernel_size=11, strides=1, activation='relu', kernel_initializer=KERNEL_INITIALIZER)(x_input)
    x = MaxPooling1D(pool_size=4, strides=2, padding='valid')(x)
    x = Convolution1D(filters=64, kernel_size=11, strides=1, activation='relu', kernel_initializer=KERNEL_INITIALIZER)(x)
    x = MaxPooling1D(pool_size=4, strides=2, padding='valid')(x)
    x = Convolution1D(filters=128, kernel_size=11, strides=1, activation='relu', kernel_initializer=KERNEL_INITIALIZER)(x)
    x = MaxPooling1D(pool_size=4, strides=2, padding='valid')(x)
    x = Convolution1D(filters=128, kernel_size=11, strides=1, activation='relu', kernel_initializer=KERNEL_INITIALIZER)(x)
    x = MaxPooling1D(pool_size=4, strides=2, padding='valid')(x)
    # Apply attention
    #x = attention_block(x)

    x = Flatten()(x)
    x = Dense(100, activation='relu', kernel_initializer=KERNEL_INITIALIZER)(x)
    x = Dropout(0.5)(x)
    x = Dense(50, activation='relu', kernel_initializer=KERNEL_INITIALIZER)(x)
    x = Dropout(0.5)(x)
    x = Dense(nclasses, activation='softmax', kernel_initializer=KERNEL_INITIALIZER)(x)

    model = Model(inputs=x_input, outputs=x)
    return model

def saraCnn2(input_shape, nclasses):
    x_input = Input(input_shape)    
    x = Convolution1D(filters=16, kernel_size=11, strides=1, activation='relu', kernel_initializer=KERNEL_INITIALIZER) (x_input)
    x = MaxPooling1D(pool_size=(4),padding='valid',strides=2)(x)
    x = Convolution1D(filters=32, kernel_size=11, strides=1, activation='relu', kernel_initializer=KERNEL_INITIALIZER) (x)
    x = MaxPooling1D(pool_size=(4),padding='valid',strides=2)(x)
    x = Convolution1D(filters=64, kernel_size=11, strides=1, activation='relu', kernel_initializer=KERNEL_INITIALIZER) (x)
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

def saraCNN3(input_shape, nclasses):
    x_input = Input(input_shape)
    x = Convolution1D(filters=32, kernel_size=32, strides=1, activation='relu', kernel_initializer=KERNEL_INITIALIZER, kernel_regularizer='l2') (x_input)
    x = MaxPooling1D(pool_size=(4),padding='valid')(x)
    x = Convolution1D(filters=16, kernel_size=16, strides=1, activation='relu', kernel_initializer=KERNEL_INITIALIZER, kernel_regularizer='l2') (x)
    x = MaxPooling1D(pool_size=(4),padding='valid')(x)
    x = Convolution1D(filters=16, kernel_size=8, strides=1, activation='relu', kernel_initializer=KERNEL_INITIALIZER, kernel_regularizer='l2') (x)
    x = MaxPooling1D(pool_size=(2),padding='valid')(x)
    x = Flatten()(x)
    x = Dense(100, activation = 'relu', kernel_initializer=KERNEL_INITIALIZER)(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = Dense(50, activation = 'relu', kernel_initializer=KERNEL_INITIALIZER)(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = Dense(nclasses, activation = 'softmax', kernel_initializer=KERNEL_INITIALIZER)(x)
    #create model
    model = Model(inputs = x_input, outputs = x , name = 'saraCnn3')
    return model

def modelloLorenzo(input_shape,nclasses):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(32, 32, activation='relu', input_shape=input_shape, kernel_regularizer='l2'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=4))
    model.add(tf.keras.layers.BatchNormalization(axis=1))
    model.add(tf.keras.layers.Conv1D(16, 16, activation='relu', kernel_regularizer='l2'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=4))
    model.add(tf.keras.layers.BatchNormalization(axis=1))
    model.add(tf.keras.layers.Conv1D(16, 8, activation='relu', kernel_regularizer='l2'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model.add(tf.keras.layers.BatchNormalization(axis=1))
    model.add(tf.keras.layers.GRU(16, dropout=0.5))
    model.add(tf.keras.layers.Dense(nclasses, activation='sigmoid'))
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

def gruSara4(input_shape,nclasses):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(16, 32, activation='relu', input_shape=input_shape, kernel_regularizer='l2'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=4))
    model.add(tf.keras.layers.BatchNormalization(axis=1))
    model.add(tf.keras.layers.Conv1D(32, 16, activation='relu', kernel_regularizer='l2'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=4))
    model.add(tf.keras.layers.BatchNormalization(axis=1))
    model.add(tf.keras.layers.Conv1D(64, 8, activation='relu', kernel_regularizer='l2'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model.add(tf.keras.layers.BatchNormalization(axis=1))
    model.add(tf.keras.layers.GRU(32, dropout=0.5))
    model.add(tf.keras.layers.Dense(nclasses, activation='sigmoid'))
    return model

# create train-validation splits for k-fold cross validation
dataPaths = glob(os.path.join(f'{DATASETDIR}','target','*.npz'))+ glob(os.path.join(f'{DATASETDIR}','controls','*.npz'))
#X_test = glob(os.path.join(f'{DATASETDIR}/test','*','*.npy'))
#y_test = createLabels(X_test,CLASSES)
print(len(dataPaths))
groups = get_id(dataPaths)
groups = list(np.unique(groups))
if USE_LOSO:
  K = len(groups)

#sub_ids = get_id(dataPaths)
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
  class1_train = np.sum(y_train)
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
  model = gruSara4(input_shape = INPUT_SHAPE, nclasses = len(CLASSES))
  model.summary()
  model.compile(optimizer=OPTIMIZER,loss=LOSSFUNCTION, metrics=['accuracy'])


  # CREATE CALLBACKS
  tensorboard = LRTensorBoard(log_dir=os.path.join(LOGDIR,f'{i}fold'),histogram_freq=0)
  checkpointPath = os.makedirs(os.path.join(WEIGHTSDIR,f"{i}fold"),exist_ok=True)
  model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(os.path.join(WEIGHTSDIR,f"{i}fold"), f'{i}fold_best.keras'),
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
  try:
      history = model.fit(ds_train,
                epochs = EPOCHS,
                verbose = 1,
                validation_data = ds_valid,
                callbacks=[tensorboard, model_checkpoint])
      
      np.savetxt(os.path.join(LOGDIR,f'{i}fold',f'X_TRAIN{i}.txt'), X_train, fmt='%s')
      np.savetxt(os.path.join(LOGDIR,f'{i}fold',f'X_val{i}.txt'), X_val, fmt='%s')
      logging.info(f'Model training completed for fold {i}')
  except Exception as e:
     logging.error(f'Error during training on fold {i}: {e}')
  #model.save(os.path.join(MODELDIR,MODELNAME,f'{i}fold_end.h5'))

  #### TESTING ####
  print('start testing...')
  try:
    modelB = tf.keras.models.load_model(os.path.join(os.path.join(WEIGHTSDIR, f"{i}fold"), f'{i}fold_best.keras'))
    logging.info(f'Model loaded successfully for fold {i}')
    score = modelB.evaluate(ds_valid, verbose=1, callbacks=[tensorboard])
    print('Val loss:', score[0])
    print('Val accuracy:', score[1])
    accuracies.append(score[1])
    losses.append(score[0])
  except Exception as e:
    logging.error(f'Error loading model for fold {i}: {e}')


np.savetxt(os.path.join(MODELDIR,MODELNAME,'accuracies.txt'), accuracies)
np.savetxt(os.path.join(MODELDIR,MODELNAME,'losses.txt'), losses)
print(f'Mean accuracy is : {np.mean(accuracies)}')
print(f'Mean loss is : {np.mean(losses)}')
