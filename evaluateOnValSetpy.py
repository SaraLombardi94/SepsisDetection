# -*- coding: utf-8 -*-
"""
Created on Fri May  3 17:18:45 2024

@author: Sara
"""
import random
import os
import numpy as np
import tensorflow.keras
import tensorflow as tf
from glob import glob
import sklearn
from tensorflow.keras.utils import to_categorical

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
import os
from glob import glob
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
import pandas as pd
import seaborn as sns

FS=125
CLASSES = ['control','microcirculation']
N_FOLD = 5
BATCH_SIZE = 32
N_CLASSES = 2
NORMRANGE = (-1,1)
NORMALIZE = True
WINDOW_LENGTH = FS * 30 * 1 
USE_WINDOWS = True
modelDir = r"C:\Users\Utente\Desktop\wetransfer_controls-microcirculation_2024-04-23_1250\controls-microcirculation\tf_bilanciato\modelli\5fold_dsTM2min30_gru_plus_adjusted_64_mse_bs32_lre0.0001_windows3750onset_jitterFalse_ep150_1\weights"
valtxtDir = r"C:\Users\Utente\Desktop\wetransfer_controls-microcirculation_2024-04-23_1250\controls-microcirculation\tf_bilanciato\modelli\5fold_dsTM2min30_gru_plus_adjusted_64_mse_bs32_lre0.0001_windows3750onset_jitterFalse_ep150_1\logs"

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

def get_id(data_path):
  sub_ids=[]
  for item in data_path:
    file_name = item.split(os.path.sep)[-1].rstrip('.npz')
    sub_id = file_name.split('_')[0]
    if not sub_id.startswith('p'):
      raise Exception(f'Subject name has to start with "p", for example pleth0. Found {sub_id}')
    sub_ids.append(sub_id)
  return sub_ids


def create_dataset(X_val, y_val):

  ds_valid = tf.data.Dataset.from_tensor_slices((X_val, y_val))
  

  # Validation dataset
  ds_valid = ds_valid.map(lambda filepath, label: tf.numpy_function(
        load_and_select_window, [filepath, label], (tf.double, tf.float32)))
  
  ds_valid = ds_valid.cache()

  ds_valid = ds_valid.batch(BATCH_SIZE)
  ds_valid = ds_valid.map(fix_shape)
  return ds_valid

def create_confusion_matrix(y,y_pred,classLabel):
  cm = confusion_matrix(y, y_pred)
  df_cm = pd.DataFrame(cm, classLabel, classLabel)
  plt.figure(figsize = (10,6))
  conf = sns.heatmap(df_cm, annot=True, fmt='g', cmap='Blues', annot_kws={"color": "black"})
  conf.set_xlabel('Prediction')
  conf.set_ylabel('True')
  plt.show()
  plt.close()
  print(cm)

  TP = cm[0][0]
  FN = cm[0][1]
  FP = cm[1][0]
  TN = cm[1][1]

  # Sensitivity, hit rate, recall, or true positive rate
  TPR = TP/(TP+FN)
  # Specificity or true negative rate
  TNR = TN/(TN+FP) 
  # Precision or positive predictive value
  PPV = TP/(TP+FP)
  # Negative predictive value
  NPV = TN/(TN+FN)
  # Fall out or false positive rate
  FPR = FP/(FP+TN)
  # False negative rate
  FNR = FN/(TP+FN)
  # Overall accuracy
  ACC = (TP+TN)/(TP+FP+FN+TN)
  print(f'Accuracy is :{100*ACC:0.2f}%')
  print(f'Sensitivity is : {100*TPR:0.2f}%')
  print(f'Specificity is {100*TNR:0.2f}%')  
  print(f'Precision is {100*PPV:0.2f}%')
  return cm


def plot_saliency_map(model, x, y_true, correct):
    x_tensor = tf.convert_to_tensor(x.reshape(1, *x.shape), dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x_tensor)
        y_pred = model(x_tensor, training=False)
        loss = tf.keras.losses.categorical_crossentropy(y_true.reshape(1, -1), y_pred)
    
    gradients = tape.gradient(loss, x_tensor)
    gradients = tf.reduce_mean(tf.abs(gradients), axis=0).numpy().flatten()
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(x.flatten())
    plt.title('Original Signal')
    
    plt.subplot(1, 2, 2)
    plt.plot(gradients, color='green' if correct else 'red')
    plt.title('Saliency Map')
    plt.show()




for fold in range(N_FOLD):
    modelPath = os.path.join(modelDir,f"{fold}fold.keras")
    txtData = os.path.join(valtxtDir,f"{fold}fold",f"X_val{fold}.txt")
    val_paths = np.loadtxt(txtData,dtype="str",comments=None)
    y_val = createLabels(val_paths,CLASSES)
    ds_valid = create_dataset(val_paths,y_val)
    model = tf.keras.models.load_model(modelPath)
    y_pred = model.predict(ds_valid, verbose=1)
    y_pred_class = np.argmax(y_pred,axis=1)
    cm = create_confusion_matrix(y_val,y_pred_class,CLASSES)
    indexes = [i for i in range(len(y_val)) if y_val[i] != y_pred_class[i]]
    wrong_predictions = val_paths[indexes]
    print(wrong_predictions)
    
    # Visualizza la saliency map per tutti i segnali
    for idx, path in enumerate(val_paths):
        x, _ = load_and_select_window(path, y_val[idx])  # Carica il segnale
        y_true = to_categorical([y_val[idx]], num_classes=len(CLASSES))  # Converte in one-hot
        correct = y_val[idx] == y_pred_class[idx]  # Determina se la classificazione è stata corretta
        plot_saliency_map(model, x, y_true, correct)
    
    