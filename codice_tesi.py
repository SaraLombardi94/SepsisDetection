# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 08:46:41 2024

@author: paolo
"""
import numpy as np
import pandas as pd 
import os


OutDir = r"C:\Users\Utente\Desktop\Repository GitHub\tesi magistrale\data"

df_sani = pd.read_excel(r"C:/Users/Utente/Desktop/Repository GitHub/tesi magistrale/data/features_segnali_controlli.xlsx")
df_patologici = pd.read_excel(r"C:/Users/Utente/Desktop/Repository GitHub/tesi magistrale/data/features_segnali_sepsi.xlsx")

df_sani = df_sani.assign(label=0)
df_patologici = df_patologici.assign(label=1)

df_merged_init = pd.concat([df_sani,df_patologici], axis=0)
#print(df_merged)
# Salva il DataFrame aggiornato nel file Excel
df_merged_init.to_excel(r"C:/Users/Utente/Desktop/Repository GitHub/tesi magistrale/data/features_segnali_merged.xlsx", index=False)
df_merged = pd.read_excel(r"C:/Users/Utente/Desktop/Repository GitHub/tesi magistrale/data/features_segnali_merged.xlsx")
'''adesso devo eliminare gli outliers'''

# Estrai i primi 7 caratteri dalla prima cella di ogni riga perche sono quelli identificativi di un paziente
def extract_patient_id(row):
    return row[0][:7]

# Crea una nuova colonna nel DataFrame per i primi 7 caratteri
df_merged['Patient_ID'] = df_merged.apply(extract_patient_id, axis=1)
#print(df_merged)


# Sposta la nuova colonna come prima colonna
cols = list(df_merged.columns)
cols = ['Patient_ID'] + cols[:-1]
df_merged = df_merged[cols]

# Salva il DataFrame modificato in un nuovo file Excel
df_merged.to_excel(r"C:/Users/Utente/Desktop/Repository GitHub/tesi magistrale/data/features_segnali_merged.xlsx", index=False)


'''ora calcolo gli outliers per ogni paziente'''
def find_outliers(dataset):
    outliers = []

    for patient_id, patient_data in dataset.groupby('Patient_ID'):
        # Seleziona solo le colonne numeriche per questo paziente
        numeric_columns = patient_data.select_dtypes(include=np.number)

        # Calcola i range interquartili per ciascuna feature per questo paziente
        q1 = numeric_columns.quantile(0.15)
        q3 = numeric_columns.quantile(0.85)
        iqr = q3 - q1

        # Calcola i limiti inferiore e superiore per individuare gli outlier
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Identifica gli outlier per ciascuna feature e aggiungili alla lista degli outlier
        outlier_mask = ((numeric_columns < lower_bound) | (numeric_columns > upper_bound)).any(axis=1)
        outliers.extend(patient_data[outlier_mask].index.tolist())

    return outliers


# Utilizzo dell'implementazione
outlier_indices = find_outliers(df_merged)


'''
appena fatto cio posso andare ad eliminare le righe che contengono gli outliers con tale codice

'''

# Utilizza il metodo drop per eliminare le righe con gli indici specificati
df_no_outliers = df_merged.drop(outlier_indices)
# Ora 'df_no_outliers' conterrà le righe originali meno quelle con gli indici specificati


# Salva il DataFrame in un file Excel
df_no_outliers.to_excel(r"C:\Users\Utente\Desktop\Repository GitHub\tesi magistrale\data\features_segnali_merged_nooutliers.xlsx", index=False)
# Imposta index=False se non vuoi includere gli indici nel file Excel

#elimina pazienti che non hanno almeno 70 segnali e quelli dopo i 70 segnali
#prima di tutto conto quanti pazienti ci sono inizialmente  per vedere effettivamente quanti ne vengono eliminati
#specifica la colonna di interesse
colonna_interessata = 'Patient_ID'
# Conta quanti valori unici ci sono
valori_unici = df_no_outliers[colonna_interessata].nunique()


# Controllo per ogni riga se il valore della prima colonna è uguale al valore della riga precedente
dup_mask = df_no_outliers[df_no_outliers.columns[0]] == df_no_outliers[df_no_outliers.columns[0]].shift()

# Applico un'altra maschera per identificare dove inizia una sequenza di più di 70 duplicati consecutivi
dup_mask = dup_mask.groupby((dup_mask != dup_mask.shift()).cumsum()).cumsum() > 70

# Applico la maschera al dataframe
df_no_outliers = df_no_outliers[~dup_mask]




# Controllo per ogni riga se il valore della prima colonna è uguale al valore della riga precedente
dup_mask_2 = df_no_outliers[df_no_outliers.columns[0]] == df_no_outliers[df_no_outliers.columns[0]].shift()

# Applico un'altra maschera per identificare le righe che hanno meno di 70 duplicati consecutivi
mask_count = dup_mask_2.groupby((dup_mask_2 != dup_mask_2.shift()).cumsum()).transform('count')
dup_mask_2 = mask_count >= 70

# Applico la maschera al dataframe
df_no_outliers = df_no_outliers[dup_mask_2]


# Costruisco il percorso completo del file di output
output_file = os.path.join(OutDir, "features_segnali_merged_nooutliers_100sigxpatient_nolessthan100.xlsx")

# Salvo il dataframe modificato in un nuovo file excel
df_no_outliers.to_excel(output_file, index=False)













