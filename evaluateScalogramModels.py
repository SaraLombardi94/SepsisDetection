# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 10:59:50 2024

@author: Sara
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May  3 17:18:45 2024
@author: Sara
"""
import random
import os
import pandas as pd
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
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.initializers import GlorotUniform
import ssqueezepy as sp
from scipy.ndimage import zoom
from dnnImageModels import *
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
import seaborn as sns


outExcelPath = r"D:/phD_Sara/models/results/resultsScalogram.xlsx"
FS = 125
N_CLASSES = 2 # control, sepsis 
LR = 1e-4
BATCH_SIZE = 32  #aggiornamento dei pesi della rete 
EPOCHS = 200
N_FOLD = 5
NSAMPLES = FS*30
WINDOW_LENGTH = FS * 30 * 1 
NORMRANGE = (-1,1)
CLASSES = ['controls','target']
NORMALIZE = True
USE_SHUFFLE = True
USE_WINDOWS = True
USE_JITTER = False
USE_LOSO = False
USE_SCALOGRAM = True
RESIZE_IMG = True
SAVE_RESULT = True
mainDir = r"D:\phD_Sara\models\bestModelImage"


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
    Wx, scales = sp.cwt(signal_data, 'morlet', scales='log', derivative=False, nv=32)
    Wx = np.abs(Wx)
    # Convert to image
    Wx_rgb = Wx[:, :, np.newaxis]
    Wx_rgb = np.concatenate([Wx_rgb]*3, axis=-1)
    Wx_rgb = Wx_rgb.astype(np.float32)
    #tf.print(f"{Wx_rgb.shape}_conv")

    if RESIZE_IMG:
        resized_image = resize_array(Wx_rgb, 224, 224)
        return resized_image, y
    else:    
        return Wx_rgb, y

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

def get_id(data_path):
  sub_ids=[]
  for item in data_path:
    file_name = item.split(os.path.sep)[-1].rstrip('.npz')
    if "_" in file_name:
        sub_id = file_name.split('_')[0]
    elif "-" in file_name:
        sub_id = file_name.split('-')[0]
    if not sub_id.startswith('p'):
      raise Exception(f'Subject name has to start with "p", for example pleth0. Found {sub_id}')
    sub_ids.append(sub_id)
  return sub_ids

def create_dataset(X_val, y_val):
  ds_valid = tf.data.Dataset.from_tensor_slices((X_val, y_val))

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
  
  return ds_valid

def create_confusion_matrix(y,y_pred,classLabel,title):
  cm = confusion_matrix(y, y_pred)
  df_cm = pd.DataFrame(cm, classLabel, classLabel)
  plt.figure(figsize = (10,6))
  conf = sns.heatmap(df_cm, annot=True, fmt='g', cmap='Blues', annot_kws={"color": "black"})
  conf.set_xlabel('Prediction')
  conf.set_ylabel('True')
  plt.title(title)
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
  return cm, ACC, TPR, TNR

for dnn in glob(os.path.join(mainDir,"*")):
    modelDir = os.path.join(dnn,"weights")
    valtxtDir = os.path.join(dnn,"logs")
    #modelDir = r"D:\phD_Sara\models\control_target\signals\k5_gruSara3bis_plus_attention_lr5e-06__BATCH64_ep50020240528_120804\weights"
    #valtxtDir = r"D:\phD_Sara\models\control_target\signals\k5_gruSara3bis_plus_attention_lr5e-06__BATCH64_ep50020240528_120804\logs"
    results = []
    accuracies, sensitivities, specificities = [],[],[]
    for fold in range(N_FOLD):
        modelPath = os.path.join(modelDir,f"{fold}fold.keras")
        txtData = os.path.join(valtxtDir,f"{fold}fold",f"X_val{fold}.txt")
        val_paths = np.loadtxt(txtData,dtype="str",comments=None)
        y_val = createLabels(val_paths,CLASSES)
        ds_valid = create_dataset(val_paths,y_val)
        try:
            model = tf.keras.models.load_model(modelPath)
            #model.summary()
        except:
           print(modelPath)
           continue
        y_pred = model.predict(ds_valid, verbose=1)
        y_pred_class = np.argmax(y_pred,axis=1)
        cm, acc, sens, spec = create_confusion_matrix(y_val,y_pred_class,CLASSES,f"{fold}_{modelDir.split(os.path.sep)[-2]}")
        accuracies.append(acc), sensitivities.append(sens), specificities.append(spec)
        indexes = [i for i in range(len(y_val)) if y_val[i] != y_pred_class[i]]
        wrong_predictions = val_paths[indexes]
        print(wrong_predictions)
        
    print(f"Model {modelDir.split(os.path.sep)[-2]}\n")
    print(f"Mean Accuracies: {np.mean(np.array(accuracies))}\n ")
    print(f"Mean Sensitivity: {np.mean(np.array(sensitivities))}\n")
    print(f"Mean Specificity: {np.mean(np.array(specificities))}\n")
    
    results.append({
        "Model": modelDir.split(os.path.sep)[-2],
        "Accuracy": np.mean(np.array(accuracies)),
        "Sensitivity": np.mean(np.array(sensitivities)),
        "Specificity": np.mean(np.array(specificities))
        })
    
    if SAVE_RESULT:
        # results.append({
        #     "Model": modelDir.split(os.path.sep)[-2],
        #     "Accuracy": np.mean(np.array(accuracies)),
        #     "Sensitivity": np.mean(np.array(sensitivities)),
        #     "Specificity": np.mean(np.array(specificities))
        #     })
        # Leggi il file Excel esistente
        df = pd.read_excel(outExcelPath)
        
        # Aggiungi le nuove righe al DataFrame
        new_results_df = pd.DataFrame(results)
        df = pd.concat([df, new_results_df], ignore_index=True)
        
        # Salva il DataFrame aggiornato nel file Excel
        df.to_excel(outExcelPath, index=False)
        
        
        
        
        