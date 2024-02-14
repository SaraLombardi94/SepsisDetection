# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 16:21:17 2024

@author: Utente
"""

# Importo le librerie necessarie 
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np

file_excel = r"D:/phD_Sara/tesiPaolo/SepsisDetection/data/features_segnali_controlli_100sigxpatient_nopatientwithlessthan100sig_nooutliers.xlsx"

df = pd.read_excel(file_excel)
data = df.to_numpy()


'''
#se voglio plottare le features di tutti i pazienti 
def plot_histograms(dataset):
    num_patients = dataset.shape[0] // 110
    num_features = dataset.shape[1]

    for i in range(num_patients):
        patient_data = dataset[i*110 : (i+1)*110]
        fig, axs = plt.subplots(num_features, figsize=(8, 6*num_features))

        for j in range(num_features):
            feature_data = patient_data[:, j]
            axs[j].hist(feature_data, bins=20, color='skyblue', edgecolor='black')
            axs[j].set_title(f'Patient {i+1} - Feature {j+1} Histogram')
            axs[j].set_xlabel('Value')
            axs[j].set_ylabel('Frequency')

        plt.tight_layout()
        plt.show()

# Assuming your dataset is a numpy array called 'data'
# The shape of 'data' should be (num_patients * 110, num_features)
# Where num_patients is the number of patients and num_features is the number of features
# You can load your data into 'data' and then call the function like this:
# plot_histograms(data)

plot_histograms(data)
'''


#se volessi plottare solo per i primi 4 pazienti
def plot_histograms(dataset):
    num_patients = min(dataset.shape[0] // 110, 4)  # Considera solo i primi 4 pazienti
    num_features = dataset.shape[1]

    for i in range(num_patients):
        patient_data = dataset[i*110 : (i+1)*110]
        fig, axs = plt.subplots(num_features, figsize=(8, 6*num_features))

        for j in range(num_features):
            feature_data = patient_data[:, j]
            axs[j].hist(feature_data, bins=20, color='skyblue', edgecolor='black')
            axs[j].set_title(f'Patient {i+1} - Feature {j+1} Histogram')
            axs[j].set_xlabel('Value')
            axs[j].set_ylabel('Frequency')

        plt.tight_layout()
        plt.show()
#plot_histograms(data)


#se voglio plottare solo alcune features che ritengo importanti per capire quali sono gli ouotliers
def plot_histograms_onlysomefeatures(dataset, columns_to_plot):
    num_patients = min(dataset.shape[0] // 110, 4) #se voglio applicarlo a tutti i pazienti basta levare min e il 4, in questo modo lo si fa solo per i primi 4 pazienti
    num_features = len(columns_to_plot)

    for i in range(num_patients):
        patient_data = dataset[i*110 : (i+1)*110]
        fig, axs = plt.subplots(num_features, figsize=(8, 6*num_features))

        for j, col in enumerate(columns_to_plot):
            feature_data = patient_data[:, df.columns.get_loc(col)]  # Ottieni l'indice della colonna
            axs[j].hist(feature_data, bins=20, color='skyblue', edgecolor='black')
            axs[j].set_title(f'Patient {i+1} - {col} Histogram')
            axs[j].set_xlabel('Value')
            axs[j].set_ylabel('Frequency')

        plt.tight_layout()
        plt.show()
# Estrapola le colonne desiderate
colonne_da_estrarre = ['media', 'std', 'frequenza VLF dove ho il picco', 'frequenza LF dove ho il picco', 'frequenza HF dove ho il picco']  # sostituisci con i nomi delle colonne che desideri estrarre
# Chiamare la funzione plot_histograms con il dataframe 'df' e le colonne specificate
plot_histograms_onlysomefeatures(data, colonne_da_estrarre)

