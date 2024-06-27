# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 09:57:18 2024

@author: Sara
"""
import tensorflow as tf
import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Add, Dense, BatchNormalization, Activation, Dropout, Flatten, GlobalAveragePooling2D, AveragePooling2D
from tensorflow.keras.layers import Convolution1D, ZeroPadding1D, MaxPooling1D, AveragePooling1D, Conv2D, MaxPooling2D, concatenate
from tensorflow.keras.layers import LSTM, TimeDistributed

DROPOUT_RATE = 0.3


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
    return model



def cnnModel2D_0(input_shape, nclasses):
    model = Sequential()

    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(nclasses, activation='softmax'))
    
    return model


def cnnModel2D_1(input_shape, nclasses):
    model = Sequential()

    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(nclasses, activation='softmax'))
    
    return model

def Inception_block(input_layer, f1, f2_conv1, f2_conv3, f3_conv1, f3_conv5, f4): 
  # Input: 
  # - f1: number of filters of the 1x1 convolutional layer in the first path
  # - f2_conv1, f2_conv3 are number of filters corresponding to the 1x1 and 3x3 convolutional layers in the second path
  # - f3_conv1, f3_conv5 are the number of filters corresponding to the 1x1 and 5x5  convolutional layer in the third path
  # - f4: number of filters of the 1x1 convolutional layer in the fourth path

  # 1st path:
  path1 = Conv2D(filters=f1, kernel_size = (1,1), padding = 'same', activation = 'relu')(input_layer)

  # 2nd path
  path2 = Conv2D(filters = f2_conv1, kernel_size = (1,1), padding = 'same', activation = 'relu')(input_layer)
  path2 = Conv2D(filters = f2_conv3, kernel_size = (3,3), padding = 'same', activation = 'relu')(path2)

  # 3rd path
  path3 = Conv2D(filters = f3_conv1, kernel_size = (1,1), padding = 'same', activation = 'relu')(input_layer)
  path3 = Conv2D(filters = f3_conv5, kernel_size = (5,5), padding = 'same', activation = 'relu')(path3)

  # 4th path
  path4 = MaxPooling2D((3,3), strides= (1,1), padding = 'same')(input_layer)
  path4 = Conv2D(filters = f4, kernel_size = (1,1), padding = 'same', activation = 'relu')(path4)

  output_layer = concatenate([path1, path2, path3, path4], axis = -1)

  return output_layer


def GoogLeNet():
  # input layer 
  input_layer = Input(shape = (224, 224, 3))

  # convolutional layer: filters = 64, kernel_size = (7,7), strides = 2
  X = Conv2D(filters = 64, kernel_size = (7,7), strides = 2, padding = 'valid', activation = 'relu')(input_layer)

  # max-pooling layer: pool_size = (3,3), strides = 2
  X = MaxPooling2D(pool_size = (3,3), strides = 2)(X)

  # convolutional layer: filters = 64, strides = 1
  X = Conv2D(filters = 64, kernel_size = (1,1), strides = 1, padding = 'same', activation = 'relu')(X)

  # convolutional layer: filters = 192, kernel_size = (3,3)
  X = Conv2D(filters = 192, kernel_size = (3,3), padding = 'same', activation = 'relu')(X)

  # max-pooling layer: pool_size = (3,3), strides = 2
  X = MaxPooling2D(pool_size= (3,3), strides = 2)(X)

  # 1st Inception block
  X = Inception_block(X, f1 = 64, f2_conv1 = 96, f2_conv3 = 128, f3_conv1 = 16, f3_conv5 = 32, f4 = 32)

  # 2nd Inception block
  X = Inception_block(X, f1 = 128, f2_conv1 = 128, f2_conv3 = 192, f3_conv1 = 32, f3_conv5 = 96, f4 = 64)

  # max-pooling layer: pool_size = (3,3), strides = 2
  X = MaxPooling2D(pool_size= (3,3), strides = 2)(X)

  # 3rd Inception block
  X = Inception_block(X, f1 = 192, f2_conv1 = 96, f2_conv3 = 208, f3_conv1 = 16, f3_conv5 = 48, f4 = 64)

  # Extra network 1:
  X1 = AveragePooling2D(pool_size = (5,5), strides = 3)(X)
  X1 = Conv2D(filters = 128, kernel_size = (1,1), padding = 'same', activation = 'relu')(X1)
  X1 = Flatten()(X1)
  X1 = Dense(1024, activation = 'relu')(X1)
  X1 = Dropout(0.7)(X1)
  X1 = Dense(5, activation = 'softmax')(X1)

  
  # 4th Inception block
  X = Inception_block(X, f1 = 160, f2_conv1 = 112, f2_conv3 = 224, f3_conv1 = 24, f3_conv5 = 64, f4 = 64)

  # 5th Inception block
  X = Inception_block(X, f1 = 128, f2_conv1 = 128, f2_conv3 = 256, f3_conv1 = 24, f3_conv5 = 64, f4 = 64)

  # 6th Inception block
  X = Inception_block(X, f1 = 112, f2_conv1 = 144, f2_conv3 = 288, f3_conv1 = 32, f3_conv5 = 64, f4 = 64)

  # Extra network 2:
  X2 = AveragePooling2D(pool_size = (5,5), strides = 3)(X)
  X2 = Conv2D(filters = 128, kernel_size = (1,1), padding = 'same', activation = 'relu')(X2)
  X2 = Flatten()(X2)
  X2 = Dense(1024, activation = 'relu')(X2)
  X2 = Dropout(0.7)(X2)
  X2 = Dense(1000, activation = 'softmax')(X2)
  
  
  # 7th Inception block
  X = Inception_block(X, f1 = 256, f2_conv1 = 160, f2_conv3 = 320, f3_conv1 = 32, 
                      f3_conv5 = 128, f4 = 128)

  # max-pooling layer: pool_size = (3,3), strides = 2
  X = MaxPooling2D(pool_size = (3,3), strides = 2)(X)

  # 8th Inception block
  X = Inception_block(X, f1 = 256, f2_conv1 = 160, f2_conv3 = 320, f3_conv1 = 32, f3_conv5 = 128, f4 = 128)

  # 9th Inception block
  X = Inception_block(X, f1 = 384, f2_conv1 = 192, f2_conv3 = 384, f3_conv1 = 48, f3_conv5 = 128, f4 = 128)

  # Global Average pooling layer 
  X = GlobalAveragePooling2D(name = 'GAPL')(X)

  # Dropoutlayer 
  X = Dropout(0.4)(X)

  # output layer 
  X = Dense(1000, activation = 'softmax')(X)
  
  # model
  model = Model(input_layer, [X, X1, X2], name = 'GoogLeNet')

  return model

def vgg19(input_shape, nclasses):
    base_model = tf.keras.applications.VGG19( 
                        weights ='imagenet',
                        include_top = False,
                        input_shape = input_shape)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(nclasses, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def modello_scalogrammi_vgg16(input_shape, nclasses, unfrozen_layers=15):
    base_model = tf.keras.applications.VGG16(
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
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = Dropout(DROPOUT_RATE)(x)  # Assuming DROPOUT_RATE is 0.5
    outputs = Dense(nclasses, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model

def InceptionV3(input_shape, nclasses, unfrozen_layers=15):
    base_model = tf.keras.applications.InceptionV3(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=input_shape,
    pooling=None
    )
    # Freeze all layers first
    base_model.trainable = False

    # Unfreeze the last 'unfrozen_layers' layers
    for layer in base_model.layers[-unfrozen_layers:]:
        layer.trainable = True

    # x = base_model.output
    # x = GlobalAveragePooling2D()(x)
    # x = Dense(128, activation='relu')(x)
    # x = Dropout(DROPOUT_RATE)(x)
    # outputs = Dense(nclasses, activation='softmax')(x)
    # model = Model(inputs=base_model.input, outputs=outputs)

    inputs = Input(shape=input_shape)
    x = base_model(inputs,training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(DROPOUT_RATE)(x)
    outputs = Dense(nclasses, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model 

def InceptionResNetV2(input_shape, nclasses, unfrozen_layers=15):
    base_model = tf.keras.applications.InceptionResNetV2(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        )  
    # Freeze all layers first
    base_model.trainable = True

    # # Unfreeze the last 'unfrozen_layers' layers
    # for layer in base_model.layers[-unfrozen_layers:]:
    #     layer.trainable = True

    #inputs = Input(shape=input_shape)
    x = base_model(input_shape=input_shape)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(DROPOUT_RATE)(x)
    outputs = Dense(nclasses, activation='softmax')(x)
    model = Model(base_model.input, outputs)
    return model