# -*- coding: utf-8 -*-
"""
Created on Tue May 28 15:05:11 2024

@author: Sara
"""

import random
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras
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
from tensorflow.keras.layers import LSTM, TimeDistributed, Attention
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
import matplotlib.pyplot as plt
import keras.backend as K
import logging
from tensorflow.keras.layers import Layer
import datetime
from tensorflow.keras.optimizers.legacy import Adam
import tensorflow_addons as tfa
# DEFINE MODELS
DROPOUT_RATE = 0.3
RANDOM_STATE = 12
BUFFER_SHUFFLING_SIZE = 180
KERNEL_INITIALIZER='glorot_uniform'

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
    attention_probs = Dense(inputs.shape[-1], activation='softmax')(inputs)
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


def saraCnn1_att(input_shape, nclasses):
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
    x = attention_block(x)

    x = Flatten()(x)
    x = Dense(100, activation='relu', kernel_initializer=KERNEL_INITIALIZER)(x)
    x = Dropout(0.5)(x)
    x = Dense(50, activation='relu', kernel_initializer=KERNEL_INITIALIZER)(x)
    x = Dropout(0.5)(x)
    x = Dense(nclasses, activation='softmax', kernel_initializer=KERNEL_INITIALIZER)(x)

    model = Model(inputs=x_input, outputs=x)
    return model

def saraCnn2_att(input_shape, nclasses):
    x_input = Input(input_shape)    
    x = Convolution1D(filters=16, kernel_size=11, strides=1, activation='relu', kernel_initializer=KERNEL_INITIALIZER) (x_input)
    x = MaxPooling1D(pool_size=(4),padding='valid',strides=2)(x)
    x = Convolution1D(filters=32, kernel_size=11, strides=1, activation='relu', kernel_initializer=KERNEL_INITIALIZER) (x)
    x = MaxPooling1D(pool_size=(4),padding='valid',strides=2)(x)
    x = Convolution1D(filters=64, kernel_size=11, strides=1, activation='relu', kernel_initializer=KERNEL_INITIALIZER) (x)
    x = MaxPooling1D(pool_size=(4),padding='valid',strides=2)(x)
    x = Convolution1D(filters=128, kernel_size=11, strides=1, activation='relu', kernel_initializer=KERNEL_INITIALIZER) (x)
    x = MaxPooling1D(pool_size=(4),padding='valid',strides=2)(x)
    
    # Apply attention
    x = attention_block(x)
    
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
    model.add(tf.keras.layers.Dense(16, activation='sigmoid'))
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


def gruSara_plus_attention(input_shape, nclasses):
    inputs = tf.keras.Input(shape=input_shape)

    # Layer convoluzionali iniziali
    x = tf.keras.layers.Conv1D(16, 32, activation='relu', kernel_regularizer='l2')(inputs)
    x = tf.keras.layers.MaxPool1D(pool_size=4)(x)
    x = tf.keras.layers.Conv1D(32, 16, activation='relu', kernel_regularizer='l2')(x)
    x = tf.keras.layers.MaxPool1D(pool_size=4)(x)
    x = tf.keras.layers.Conv1D(64, 8, activation='relu', kernel_regularizer='l2')(x)
    x = tf.keras.layers.MaxPool1D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Nuovo layer convoluzionale aggiunto
    x = tf.keras.layers.Conv1D(64, 16, activation='relu', kernel_regularizer='l2')(x)
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
    outputs = tf.keras.layers.Dense(nclasses, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)

    return model


def gruSara2_plus_attention(input_shape, nclasses):
    inputs = tf.keras.Input(shape=input_shape)

    # Layer convoluzionali iniziali
    x = tf.keras.layers.Conv1D(32, 32, activation='relu', kernel_regularizer='l2')(inputs)
    x = tf.keras.layers.MaxPool1D(pool_size=4)(x)
    x = tf.keras.layers.Conv1D(64, 16, activation='relu', kernel_regularizer='l2')(x)
    x = tf.keras.layers.MaxPool1D(pool_size=4)(x)
    x = tf.keras.layers.Conv1D(128, 8, activation='relu', kernel_regularizer='l2')(x)
    x = tf.keras.layers.MaxPool1D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # # Nuovo layer convoluzionale aggiunto
    # x = tf.keras.layers.Conv1D(64, 16, activation='relu', kernel_regularizer='l2')(x)
    # x = tf.keras.layers.MaxPool1D(pool_size=2)(x)

    # # Aggiunta di ulteriori layer convoluzionali esistenti
    # x = tf.keras.layers.Conv1D(128, 4, activation='relu', kernel_regularizer='l2')(x)
    # x = tf.keras.layers.MaxPool1D(pool_size=2)(x)

    # Nuovo layer GRU con più unità
    x = tf.keras.layers.GRU(64, return_sequences=True)(x)

    # Layer di attenzione
    attention_output = tf.keras.layers.Attention()([x,x])
    x = tf.keras.layers.Flatten()(attention_output)
    # Layer densamente connesso finale
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(nclasses, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    return model




def gruSara4_plus_attention(input_shape, nclasses):
    inputs = tf.keras.Input(shape=input_shape)

    # Layer convoluzionali iniziali
    x = tf.keras.layers.Conv1D(16, 32, activation='relu', kernel_regularizer='l2')(inputs)
    x = tf.keras.layers.MaxPool1D(pool_size=4)(x)
    x = tf.keras.layers.Conv1D(32, 16, activation='relu', kernel_regularizer='l2')(x)
    x = tf.keras.layers.MaxPool1D(pool_size=4)(x)
    x = tf.keras.layers.Conv1D(64, 8, activation='relu', kernel_regularizer='l2')(x)
    x = tf.keras.layers.MaxPool1D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Aggiunta di ulteriori layer convoluzionali esistenti
    x = tf.keras.layers.Conv1D(128, 4, activation='relu', kernel_regularizer='l2')(x)
    x = tf.keras.layers.MaxPool1D(pool_size=2)(x)

    # Nuovo layer GRU con più unità
    x = tf.keras.layers.GRU(32, dropout=0.5, return_sequences=True)(x)

    # Layer di attenzione
    attention_output = tf.keras.layers.Attention()([x,x])
    x = tf.keras.layers.Flatten()(attention_output)
    # Layer densamente connesso finale
    outputs = tf.keras.layers.Dense(nclasses, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)

    return model

def gruSara5_plus_attention(input_shape, nclasses):
    inputs = tf.keras.Input(shape=input_shape)

    # Layer convoluzionali iniziali
    x = tf.keras.layers.Conv1D(16, 32, activation='relu', kernel_regularizer='l2')(inputs)
    x = tf.keras.layers.MaxPool1D(pool_size=4)(x)
    x = tf.keras.layers.Conv1D(32, 16, activation='relu', kernel_regularizer='l2')(x)
    x = tf.keras.layers.MaxPool1D(pool_size=4)(x)
    x = tf.keras.layers.Conv1D(64, 8, activation='relu', kernel_regularizer='l2')(x)
    x = tf.keras.layers.MaxPool1D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Aggiunta di ulteriori layer convoluzionali esistenti
    x = tf.keras.layers.Conv1D(128, 4, activation='relu', kernel_regularizer='l2')(x)
    x = tf.keras.layers.MaxPool1D(pool_size=2)(x)

    # Nuovo layer GRU con più unità
    x = tf.keras.layers.GRU(32, dropout=0.5, return_sequences=True)(x)

    # Layer di attenzione
    attention_output = tf.keras.layers.Attention()([x,x])
    x = tf.keras.layers.Flatten()(attention_output)
    # Layer densamente connesso finale
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    outputs = tf.keras.layers.Dense(nclasses, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    return model

def gruSara6_plus_attention(input_shape, nclasses):
    inputs = tf.keras.Input(shape=input_shape)
    # Layer convoluzionali iniziali
    x = tf.keras.layers.Conv1D(16, 32, activation='relu', kernel_regularizer='l2')(inputs)
    x = tf.keras.layers.MaxPool1D(pool_size=4)(x)
    x = tf.keras.layers.Conv1D(32, 16, activation='relu', kernel_regularizer='l2')(x)
    x = tf.keras.layers.MaxPool1D(pool_size=4)(x)
    x = tf.keras.layers.Conv1D(64, 8, activation='relu', kernel_regularizer='l2')(x)
    x = tf.keras.layers.MaxPool1D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # Nuovo layer convoluzionale aggiunto
    x = tf.keras.layers.Conv1D(64, 16, activation='relu', kernel_regularizer='l2')(x)
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

def gruSara6_plus_attention_chat(input_shape, nclasses):
    inputs = tf.keras.Input(shape=input_shape)
    # Layer convoluzionali iniziali
    x = tf.keras.layers.Conv1D(16, 32, activation='relu', kernel_regularizer='l2')(inputs)
    x = tf.keras.layers.MaxPool1D(pool_size=4)(x)
    x = tf.keras.layers.Conv1D(32, 16, activation='relu', kernel_regularizer='l2')(x)
    x = tf.keras.layers.MaxPool1D(pool_size=4)(x)
    x = tf.keras.layers.Conv1D(64, 8, activation='relu', kernel_regularizer='l2')(x)
    x = tf.keras.layers.MaxPool1D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # Nuovo layer convoluzionale aggiunto
    x = tf.keras.layers.Conv1D(64, 16, activation='relu', kernel_regularizer='l2')(x)
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


def gruSara7_plus_attention(input_shape, nclasses):
    inputs = tf.keras.Input(shape=input_shape)

    # Layer convoluzionali iniziali
    x = tf.keras.layers.Conv1D(16, 32, activation='relu', kernel_regularizer='l2')(inputs)
    x = tf.keras.layers.MaxPool1D(pool_size=4)(x)
    x = tf.keras.layers.Conv1D(32, 16, activation='relu', kernel_regularizer='l2')(x)
    x = tf.keras.layers.MaxPool1D(pool_size=4)(x)
    x = tf.keras.layers.Conv1D(64, 8, activation='relu', kernel_regularizer='l2')(x)
    x = tf.keras.layers.MaxPool1D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Nuovo layer convoluzionale aggiunto
    x = tf.keras.layers.Conv1D(64, 16, activation='relu', kernel_regularizer='l2')(x)
    x = tf.keras.layers.MaxPool1D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    # Aggiunta di ulteriori layer convoluzionali esistenti
    x = tf.keras.layers.Conv1D(128, 4, activation='relu', kernel_regularizer='l2')(x)
    x = tf.keras.layers.MaxPool1D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    # Nuovo layer GRU con più unità
    x = tf.keras.layers.GRU(128, dropout=0.5, return_sequences=True)(x)
    x = tf.keras.layers.GRU(64, dropout=0.5, return_sequences=True)(x)

    # Layer di attenzione
    attention_output = tf.keras.layers.Attention()([x, x])
    x = tf.keras.layers.GlobalAveragePooling1D()(attention_output)

    # Layer densamente connesso finale
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(nclasses, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    # Compilazione del modello con AdamW
    optimizer = tfa.optimizers.AdamW(learning_rate=5e-6, weight_decay=1e-5)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def gruSara9_plus_attention(input_shape, nclasses):
    inputs = tf.keras.Input(shape=input_shape)
    # Layer convoluzionali iniziali
    x = tf.keras.layers.Conv1D(16, 32, activation='relu', kernel_regularizer='l2')(inputs)
    x = tf.keras.layers.MaxPool1D(pool_size=4)(x)
    x = tf.keras.layers.Conv1D(32, 16, activation='relu', kernel_regularizer='l2')(x)
    x = tf.keras.layers.MaxPool1D(pool_size=4)(x)
    x = tf.keras.layers.Conv1D(64, 8, activation='relu', kernel_regularizer='l2')(x)
    x = tf.keras.layers.MaxPool1D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # Nuovo layer convoluzionale aggiunto
    x = tf.keras.layers.Conv1D(64, 16, activation='relu', kernel_regularizer='l2')(x)
    x = tf.keras.layers.MaxPool1D(pool_size=2)(x)
    # Aggiunta di ulteriori layer convoluzionali esistenti
    x = tf.keras.layers.Conv1D(128, 4, activation='relu', kernel_regularizer='l2')(x)
    x = tf.keras.layers.MaxPool1D(pool_size=2)(x)
    # Nuovo layer GRU con più unità
    x = tf.keras.layers.GRU(128, dropout=0.5, return_sequences=True)(x)
    # Layer di attenzione
    attention_output = tf.keras.layers.Attention()([x,x])
    x = tf.keras.layers.Flatten()(attention_output)
    # Layer densamente connesso finale
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(nclasses, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model


def gruSara2bis_plus_attention(input_shape, nclasses):
    inputs = tf.keras.Input(shape=input_shape)

    # Layer convoluzionali iniziali
    x = tf.keras.layers.Conv1D(16, 32, activation='relu', kernel_regularizer='l2')(inputs)
    x = tf.keras.layers.MaxPool1D(pool_size=4)(x)
    x = tf.keras.layers.Conv1D(32, 32, activation='relu', kernel_regularizer='l2')(inputs)
    x = tf.keras.layers.MaxPool1D(pool_size=4)(x)
    x = tf.keras.layers.Conv1D(64, 16, activation='relu', kernel_regularizer='l2')(x)
    x = tf.keras.layers.MaxPool1D(pool_size=4)(x)
    x = tf.keras.layers.Conv1D(128, 8, activation='relu', kernel_regularizer='l2')(x)
    x = tf.keras.layers.MaxPool1D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # Nuovo layer GRU con più unità
    x = tf.keras.layers.GRU(64, return_sequences=True)(x)
    # Layer di attenzione
    attention_output = tf.keras.layers.Attention()([x,x])
    x = tf.keras.layers.Flatten()(attention_output)
    # Layer densamente connesso finale
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(nclasses, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    return model


def gruSara3bis_plus_attention(input_shape, nclasses):
    inputs = tf.keras.Input(shape=input_shape)

    # Layer convoluzionali iniziali
    x = tf.keras.layers.Conv1D(16, 32, activation='relu', kernel_regularizer='l2')(inputs)
    x = tf.keras.layers.MaxPool1D(pool_size=4)(x)
    x = tf.keras.layers.Conv1D(32, 16, activation='relu', kernel_regularizer='l2')(inputs)
    x = tf.keras.layers.MaxPool1D(pool_size=4)(x)
    x = tf.keras.layers.Conv1D(64, 8, activation='relu', kernel_regularizer='l2')(x)
    x = tf.keras.layers.MaxPool1D(pool_size=2)(x)
    x = tf.keras.layers.Conv1D(128, 4, activation='relu', kernel_regularizer='l2')(x)
    x = tf.keras.layers.MaxPool1D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # Nuovo layer GRU con più unità
    x = tf.keras.layers.GRU(64, return_sequences=True)(x)
    # Layer di attenzione
    attention_output = tf.keras.layers.Attention()([x,x])
    x = tf.keras.layers.Flatten()(attention_output)
    # Layer densamente connesso finale
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(nclasses, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    return model