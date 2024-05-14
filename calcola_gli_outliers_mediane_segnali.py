# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 11:20:25 2024

@author: Utente
"""

import pandas as pd
dataset = pd.read_excel(r"C:/Users/Utente/Downloads/expModelDataset_seed4_control60.xlsx")
def remove_outliers(dataset):
    # Seleziona solo le colonne delle features, escludendo la prima colonna
    features = dataset.iloc[:, 1:]
    
    # Calcola i quartili per ciascuna feature
    q1 = features.quantile(0.25)
    q3 = features.quantile(0.75)
    
    # Calcola il range interquartile (IQR) per ciascuna feature
    iqr = q3 - q1
    
    # Calcola i limiti inferiore e superiore per individuare gli outliers
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Crea una maschera per identificare gli outliers
    outlier_mask = ((features < lower_bound) | (features > upper_bound)).any(axis=1)
    
    # Elimina le righe contenenti outliers
    dataset_filtered = dataset[~outlier_mask]
    
    return dataset_filtered

# Esempio di utilizzo
# Supponendo che 'dataset' sia il DataFrame contenente i tuoi dati
dataset_senza_outliers = remove_outliers(dataset)


# Salva il DataFrame in un file Excel
dataset_senza_outliers.to_excel(r"C:/Users/Utente/Downloads/expModelDataset_seed4_control60_nooutliers.xlsx", index=False)
# Imposta index=False se non vuoi includere gli indici nel file Excel


