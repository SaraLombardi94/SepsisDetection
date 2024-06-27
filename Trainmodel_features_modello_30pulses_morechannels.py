# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 10:07:39 2024

@author: Utente
"""

import pandas as pd
import numpy as np
import random
import os
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
LR = 1e-6
BATCH_SIZE = 32
EPOCHS = 3000
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
MODELNAME = f'k{K}_ECG_model_lr{LR}_ep{EPOCHS}_prove4'
MODELDIR = r'C:\Users\Utente\Desktop\features_modello_matematico\modelli_more_channels'
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

def attention_block(inputs):
    # a simple attention mechanism with a single dense layer
    attention_probs = Dense(inputs.shape[-1], activation='softmax', name='attention_vec')(inputs)
    attention_mul = Multiply()([inputs, attention_probs])
    return attention_mul


def saraCnn1(input_shape, nclasses):
    x_input = Input(input_shape)
    x = Convolution1D(filters=16, kernel_size=10, strides=1, activation='relu', kernel_initializer=KERNEL_INITIALIZER)(x_input)
    #x = MaxPooling1D(pool_size=2, strides=2, padding='valid')(x)
    #x = Convolution1D(filters=64, kernel_size=5, strides=1, activation='relu', kernel_initializer=KERNEL_INITIALIZER)(x)
    #x = MaxPooling1D(pool_size=2, strides=2, padding='valid')(x)
    # x = Convolution1D(filters=128, kernel_size=10, strides=1, activation='relu', kernel_initializer=KERNEL_INITIALIZER)(x)
    # x = MaxPooling1D(pool_size=2, strides=2, padding='valid')(x)
    # # x = Convolution1D(filters=128, kernel_size=11, strides=1, activation='relu', kernel_initializer=KERNEL_INITIALIZER)(x)
    # # x = MaxPooling1D(pool_size=4, strides=2, padding='valid')(x)
    # Apply attention
    #x = attention_block(x)

    x = Flatten()(x)
    x = Dense(100, activation='relu', kernel_initializer=KERNEL_INITIALIZER)(x)
    #x = Dense(100, activation='relu', kernel_initializer=KERNEL_INITIALIZER)(x)
    #x = Dropout(0.5)(x)
    # x = Dense(50, activation='relu', kernel_initializer=KERNEL_INITIALIZER)(x)
    # x = Dropout(0.5)(x)
    x = Dense(nclasses, activation='softmax', kernel_initializer=KERNEL_INITIALIZER)(x)

    model = Model(inputs=x_input, outputs=x)
    return model


def cnn_lstm_model(input_shape, nclasses):
    inputs = Input(shape=input_shape)
    
    # Strati convoluzionali
    x = Conv1D(64, 3, activation='relu', kernel_initializer='glorot_uniform')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    x = Conv1D(128, 3, activation='relu', kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    x = Conv1D(128, 3, activation='relu', kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    # Strato LSTM
    x = LSTM(100, dropout=0.5, return_sequences=True)(x)
    
    # Pooling globale
    x = GlobalAveragePooling1D()(x)
    
    # Strato densamente connesso
    x = Dense(128, activation='relu', kernel_initializer='glorot_uniform')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu', kernel_initializer='glorot_uniform')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(nclasses, activation='softmax', kernel_initializer='glorot_uniform')(x)

    model = Model(inputs, outputs)
    
    return model


def ECG_model(input_shape, nclasses):
    x_input = Input(input_shape)
    
    # Primo strato di convoluzione e pooling
    x = Conv1D(filters=32, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform')(x_input)
    x = MaxPooling1D(pool_size=2, strides=2, padding='valid')(x)
    
    # Secondo strato di convoluzione e pooling
    x = Conv1D(filters=64, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform')(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding='valid')(x)
    
    # Terzo strato di convoluzione e pooling
    x = Conv1D(filters=128, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform')(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding='valid')(x)
    
    # Normalizzazione dei batch e dropout
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # Appiattimento e strato completamente connesso
    x = Flatten()(x)
    x = Dense(100, activation='relu', kernel_initializer='glorot_uniform')(x)
    x = Dropout(0.5)(x)
    
    # Strato di output
    outputs = Dense(nclasses, activation='softmax', kernel_initializer='glorot_uniform')(x)

    model = Model(inputs=x_input, outputs=outputs)
    return model


# Leggiamo il file Excel
file_path = r'C:/Users/Utente/Desktop/features_modello_matematico/file_excels/all_features_modello_30pulses.xlsx'
df = pd.read_excel(file_path)


# # Controlliamo la struttura del dataframe
# print(df.head())
# print(df.shape)


# Numero di pazienti
num_patients = df.shape[0]

# Inizializziamo il dizionario per contenere i dati
data_dict = {}
'''
# Popoliamo il dizionario con i dati
for i in range(num_patients):
    patient_name = df.iloc[i, 0]  # Assumendo che il nome del paziente sia nella prima colonna
    patient_data = np.zeros((22, 30))
    for j in range(22):
        start_idx = 1 + j * 30
        end_idx = start_idx + 30
        patient_data[j, :] = df.iloc[i, start_idx:end_idx].values
    data_dict[patient_name] = patient_data
'''

# Popoliamo il dizionario con i dati e le labels
for i in range(num_patients):
    patient_name = df.iloc[i, 0]  # Assumendo che il nome del paziente sia nella prima colonna
    label = df.iloc[i, -1]  # Assumendo che la label sia nell'ultima colonna
    patient_data = np.zeros((22, 30))
    for j in range(22):
        start_idx = 1 + j * 30
        end_idx = start_idx + 30
        patient_data[j, :] = df.iloc[i, start_idx:end_idx].values
    data_dict[patient_name] = {'data': patient_data, 'label': label}


# Verifica del dizionario
# print(data_dict)


def get_id(data_path):
    sub_ids = []
    for item in data_path:
        sub_id = item[:7]  # Estrae i primi 7 caratteri del nome del file
        sub_ids.append(sub_id)
    return sub_ids

sub_ids = get_id(df['File_Name'])

sub_ids = set(sub_ids)

# Estrai i dati e le etichette dal dizionario
X = []
y = []
groups = []
file_names = []
for patient_name, info in data_dict.items():
    X.append(info['data'])
    y.append(info['label'])
    groups.append(patient_name[:7])  # Utilizziamo i primi 7 caratteri del nome del paziente come gruppo
    file_names.append(patient_name)
    
    
X = np.array(X)
y = np.array(y)

# Definisci il numero di split per la cross-validation
n_splits = 5

# Inizializza GroupKFold
gkf = GroupKFold(n_splits=n_splits)


accuracies = []
losses = []

# Esegui la cross-validation
for i, (train_index, val_index) in enumerate(gkf.split(X, y, groups=groups)):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    # Ottieni i file_name per i set di training e test
    train_file_names = [file_names[i] for i in train_index]
    val_file_names = [file_names[i] for i in val_index]

#print(f'Training file names: {train_file_names}')
#print(f'Test file names: {test_file_names}')

    y_train = to_categorical(y_train,num_classes=2)
    y_val = to_categorical(y_val,num_classes=2)

    INPUT_SHAPE = (22,30)
    
    model = ECG_model(input_shape = INPUT_SHAPE, nclasses = 2)
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



