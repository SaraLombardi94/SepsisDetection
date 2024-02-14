# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 16:32:17 2024

@author: Sara
"""

import pandas as pd 
import os


OutDir = r"D:\phD_Sara\tesiPaolo\SepsisDetection\data"

df_sani = pd.read_excel(r"D:/phD_Sara/tesiPaolo/SepsisDetection/data/features_segnali_controlli.xlsx")
df_patologici = pd.read_excel(r"D:/phD_Sara/tesiPaolo/SepsisDetection/data/features_segnali_sepsi.xlsx")

df_sani = df_sani.assign(label=0)
df_patologici = df_patologici.assign(label=1)

df_merged = pd.concat([df_sani,df_patologici], axis=0)
#print(df_merged)
# Salva il DataFrame aggiornato nel file Excel
df_merged.to_excel(r"D:\phD_Sara\tesiPaolo\SepsisDetection\data\features_segnali_merged.xlsx", index=False)

'''adesso devo eliminare gli outliers'''

# Estrai i primi 7 caratteri dalla prima cella di ogni riga perche sono quelli identificativi di un paziente
def extract_patient_id(row):
    return row[0][:7]

# Crea una nuova colonna nel DataFrame per i primi 7 caratteri
df_merged['Patient_ID'] = df_merged.apply(extract_patient_id, axis=1)
print(df_merged)












