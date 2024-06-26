# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 11:42:50 2024

@author: Utente
"""


import pandas as pd
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
from sklearn.preprocessing import RobustScaler
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
from tensorflow.keras.layers import Conv1D, MaxPool1D, BatchNormalization, LSTM, Dense, Dropout, Input, GlobalAveragePooling1D

#constants
FS = 125
N_CLASSES = 2 # control, sepsis
LR = 1e-4
BATCH_SIZE = 8
EPOCHS = 200
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
MODELNAME = f'k{K}_build_lstm_model_lr{LR}_ep{EPOCHS}_prove12345'
MODELDIR = r'C:\Users\Utente\Desktop\features_modello_matematico\modelli'
LOGDIR = os.path.join(MODELDIR,MODELNAME,'logs')
WEIGHTSDIR = os.path.join(MODELDIR,MODELNAME,'weights')




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
    
def robust_scaling(train_set, test_set):
    scaler = RobustScaler()
    train_set = scaler.fit_transform(train_set)
    test_set = scaler.transform(test_set)
    return train_set, test_set


def gruSara6_plus_attention(input_shape, nclasses):
    inputs = tf.keras.Input(shape=input_shape)
    # Layer convoluzionali iniziali
    x = tf.keras.layers.Conv1D(16, 3, activation='relu', kernel_regularizer='l2')(inputs)
    x = tf.keras.layers.MaxPool1D(pool_size=3)(x)
    x = tf.keras.layers.Conv1D(32, 3, activation='relu', kernel_regularizer='l2')(x)
    x = tf.keras.layers.MaxPool1D(pool_size=3)(x)
    x = tf.keras.layers.Conv1D(64, 3, activation='relu', kernel_regularizer='l2')(x)
    x = tf.keras.layers.MaxPool1D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # Nuovo layer convoluzionale aggiunto
    x = tf.keras.layers.Conv1D(64, 4, activation='relu', kernel_regularizer='l2')(x)
    x = tf.keras.layers.MaxPool1D(pool_size=2)(x)
    # Aggiunta di ulteriori layer convoluzionali esistenti
    x = tf.keras.layers.Conv1D(128, 4, activation='relu', kernel_regularizer='l2')(x)
    x = tf.keras.layers.MaxPool1D(pool_size=2)(x)
    # Nuovo layer GRU con più unità
    x = tf.keras.layers.GRU(64, dropout=0.5, return_sequences=True)(x)
    # Layer di attenzione
    attention_output = tf.keras.layers.Attention()([x,x])
    x = tf.keras.layers.Flatten()(attention_output)
    # Layer densamente connesso finale
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    outputs = tf.keras.layers.Dense(nclasses, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    return model


def simplified_gruSara6_plus_attention(input_shape, nclasses):
    inputs = tf.keras.Input(shape=input_shape)
    # Riduzione del numero dei layer convoluzionali
    x = tf.keras.layers.Conv1D(32, 3, activation='relu', kernel_regularizer='l2')(inputs)
    x = tf.keras.layers.MaxPool1D(pool_size=3)(x)
    x = tf.keras.layers.Conv1D(64, 3, activation='relu', kernel_regularizer='l2')(x)
    x = tf.keras.layers.MaxPool1D(pool_size=3)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # Layer GRU semplificato con meno unità
    x = tf.keras.layers.GRU(32, dropout=0.3, return_sequences=True)(x)
    # Layer di attenzione
    attention_output = tf.keras.layers.Attention()([x, x])
    x = tf.keras.layers.Flatten()(attention_output)
    # Layer densamente connesso finale più semplice
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    outputs = tf.keras.layers.Dense(nclasses, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    return model
def very_simplified_gruSara6_plus_attention(input_shape, nclasses):
    inputs = tf.keras.Input(shape=input_shape)
    # Utilizzo di un singolo layer convoluzionale
    x = tf.keras.layers.Conv1D(32, 3, activation='relu', kernel_regularizer='l2')(inputs)
    x = tf.keras.layers.MaxPool1D(pool_size=3)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # Utilizzo di un singolo layer GRU
    x = tf.keras.layers.GRU(16, dropout=0.2, return_sequences=True)(x)
    # Layer di attenzione semplificato
    query_value_attention_seq = tf.keras.layers.Attention()([x, x])
    query_encoding = tf.keras.layers.GlobalAveragePooling1D()(query_value_attention_seq)
    # Layer densamente connesso finale molto semplice
    x = tf.keras.layers.Dense(16, activation='relu')(query_encoding)
    outputs = tf.keras.layers.Dense(nclasses, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    return model

def gru2_very_simplified_gruSara6_plus_attention(input_shape, nclasses):
    inputs = tf.keras.Input(shape=input_shape)
    # Utilizzo di un singolo layer convoluzionale
    x = tf.keras.layers.Conv1D(32, 3, activation='relu', kernel_regularizer='l2')(inputs)
    x = tf.keras.layers.MaxPool1D(pool_size=3)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Primo strato GRU
    x = tf.keras.layers.GRU(16, dropout=0.2, return_sequences=True)(x)
    
    # Secondo strato GRU
    x = tf.keras.layers.GRU(16, dropout=0.2, return_sequences=True)(x)
    
    # Layer di attenzione semplificato
    query_value_attention_seq = tf.keras.layers.Attention()([x, x])
    query_encoding = tf.keras.layers.GlobalAveragePooling1D()(query_value_attention_seq)
    
    # Layer densamente connesso finale molto semplice
    x = tf.keras.layers.Dense(16, activation='relu')(query_encoding)
    outputs = tf.keras.layers.Dense(nclasses, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    return model

def saraCnn1(input_shape, nclasses):
    x_input = Input(input_shape)
    x = Convolution1D(filters=32, kernel_size=3, strides=1, activation='relu', kernel_initializer=KERNEL_INITIALIZER)(x_input)
    x = MaxPooling1D(pool_size=2, strides=2, padding='valid')(x)
    x = Convolution1D(filters=64, kernel_size=3, strides=1, activation='relu', kernel_initializer=KERNEL_INITIALIZER)(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding='valid')(x)
    x = Convolution1D(filters=128, kernel_size=3, strides=1, activation='relu', kernel_initializer=KERNEL_INITIALIZER)(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding='valid')(x)
    # # x = Convolution1D(filters=128, kernel_size=11, strides=1, activation='relu', kernel_initializer=KERNEL_INITIALIZER)(x)
    # # x = MaxPooling1D(pool_size=4, strides=2, padding='valid')(x)
    # # # Apply attention
    # #x = attention_block(x)

    x = Flatten()(x)
    x = Dense(100, activation='relu', kernel_initializer=KERNEL_INITIALIZER)(x)
    x = Dropout(0.5)(x)
    # x = Dense(50, activation='relu', kernel_initializer=KERNEL_INITIALIZER)(x)
    # x = Dropout(0.5)(x)
    x = Dense(nclasses, activation='softmax', kernel_initializer=KERNEL_INITIALIZER)(x)

    model = Model(inputs=x_input, outputs=x)
    return model



def improved_model(input_shape, nclasses):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Strati convoluzionali aggiuntivi
    x = tf.keras.layers.Conv1D(64, 3, activation='relu', kernel_regularizer='l2')(inputs)
    x = tf.keras.layers.MaxPool1D(pool_size=3)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.Conv1D(128, 3, activation='relu', kernel_regularizer='l2')(x)
    x = tf.keras.layers.MaxPool1D(pool_size=3)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Utilizzo di un layer Bidirectional GRU
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32, dropout=0.3, return_sequences=True))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32, dropout=0.3, return_sequences=True))(x)
    
    # Layer di attenzione
    query_value_attention_seq = tf.keras.layers.Attention()([x, x])
    query_encoding = tf.keras.layers.GlobalAveragePooling1D()(query_value_attention_seq)
    
    # Layer densamente connesso finale con dropout e regolarizzazione
    x = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer='l2')(query_encoding)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(nclasses, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    # Compilazione del modello con un learning rate scheduler
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


def build_lstm_model(input_shape, nclasses):
    inputs = Input(shape=input_shape)
    
    # Strati convoluzionali
    x = Conv1D(32, 3, activation='relu', kernel_regularizer='l2')(inputs)
    x = MaxPool1D(pool_size=3)(x)
    x = BatchNormalization()(x)
    
    x = Conv1D(64, 3, activation='relu', kernel_regularizer='l2')(x)
    x = MaxPool1D(pool_size=3)(x)
    x = BatchNormalization()(x)
    
    # Strato LSTM
    x = LSTM(64, dropout=0.5, return_sequences=True)(x)
    
    # Pooling globale
    x = GlobalAveragePooling1D()(x)
    
    # Strato densamente connesso
    x = Dense(64, activation='relu', kernel_regularizer='l2')(x)
    x = Dropout(0.4)(x)
    outputs = Dense(nclasses, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    
    return model





inputfile = r'C:/Users/Utente/Desktop/features_modello_matematico/file_excels/all_features_modello_30pulses_modified.xlsx'

dataset = pd.read_excel(inputfile)



# Extract features and labels
X = dataset.drop(columns=['File_Name', 'label'])  # Assuming these are not feature columns
y = dataset['label']

subjects = dataset['File_Name']

# set kfold crossvalidation
gkf = GroupKFold(n_splits=K)
# split is made according subject_ids
gkf.get_n_splits(X, y, subjects)
used_features = []
folds_result = {}

accuracies = []
losses = []


for i, (train_index, val_index) in enumerate(gkf.split(X, y, subjects)):
    X_train, y_train = X.iloc[train_index], y.iloc[train_index]
    X_val, y_val = X.iloc[val_index], y.iloc[val_index]
    X_train, X_val = robust_scaling(X_train, X_val)
    y_train = to_categorical(y_train,num_classes=2)
    y_val = to_categorical(y_val,num_classes=2)
    subjects_train, subjects_test = subjects.iloc[train_index], subjects.iloc[val_index]
    print(f"Fold {i}")
    INPUT_SHAPE = (660,1)
    model = build_lstm_model(input_shape = INPUT_SHAPE, nclasses = 2)
    model.summary()
    model.compile(optimizer=OPTIMIZER,loss=LOSSFUNCTION, metrics=['accuracy'])
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
        history = model.fit(x = X_train,
                            y=y_train,
                  epochs = EPOCHS,
                  verbose = 1,
                  validation_data =(X_val,y_val),
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
      score = modelB.evaluate(x=X_val,y=y_val, verbose=1, callbacks=[tensorboard])
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


