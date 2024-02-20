# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 10:42:27 2024

@author: Utente
"""

import numpy as np
import pandas as pd 
import os

OutDir = r"C:\Users\Utente\Desktop\Repository GitHub\tesi magistrale\data"
file_sani = r"C:/Users/Utente/Desktop/Repository GitHub/tesi magistrale/data/features_segnali_controlli.xlsx"
file_patologici = r"C:/Users/Utente/Desktop/Repository GitHub/tesi magistrale/data/features_segnali_sepsi.xlsx"
file_output = os.path.join(OutDir, "features_segnali_merged_nooutliers_100sigxpatient_nolessthan100_prova.xlsx")

# Carica i file excel in DataFrame e assegna una label
df_sani = pd.read_excel(file_sani).assign(label=0)
df_patologici = pd.read_excel(file_patologici).assign(label=1)

# Unisci i DataFrame
df_merged = pd.concat([df_sani, df_patologici], axis=0)

# Estrai i primi 7 caratteri dalla prima colonna per identificare i pazienti
def extract_patient_id(row):
    return row[0][:7]

df_merged['Patient_ID'] = df_merged.apply(extract_patient_id, axis=1)

# Riorganizza il DataFrame in modo che la colonna 'Patient_ID' sia la prima colonna
cols = list(df_merged.columns)
cols = ['Patient_ID'] + cols[:-1]
df_merged = df_merged[cols]

# Calcola gli outlier per ogni paziente
outliers = df_merged.groupby('Patient_ID').apply(lambda x: (x.select_dtypes(include=np.number)
                                                           .apply(lambda y: (y < y.quantile(0.15)) | (y > y.quantile(0.85)))
                                                           .any(axis=1))
                                                 .any(level=0))

# Rimuovi le righe con gli outlier
df_no_outliers = df_merged[~df_merged['Patient_ID'].isin(outliers[outliers].index)]

# Filtra i pazienti con almeno 70 segnali e rimuovi quelli dopo il 70° segnale
df_no_outliers = df_no_outliers.groupby('Patient_ID').filter(lambda x: len(x) >= 70)
df_no_outliers = df_no_outliers.groupby('Patient_ID').head(70)

# Salva il DataFrame nel file Excel
df_no_outliers.to_excel(file_output, index=False)

