# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 20:05:19 2024

@author: Utente
"""

import numpy as np
import pandas as pd

dataset = pd.read_excel(r"C:/Users/Utente/Desktop/Repository GitHub/tesi magistrale/data/features_segnali_controlli_100sigxpatient_nopatientwithlessthan100sig.xlsx")
# Supponendo che tu abbia già caricato il dataset in un DataFrame chiamato 'dataset'
OutDir = r"C:\Users\Utente\Desktop\Repository GitHub\tesi magistrale\data"

def find_outliers_by_window(dataset, window_size=100):
    outliers = []

    num_patients = dataset.shape[0] // window_size

    for i in range(num_patients):
        start_idx = i * window_size
        end_idx = min(start_idx + window_size, dataset.shape[0])
        window_data = dataset.iloc[start_idx:end_idx]

        # Seleziona solo le colonne numeriche
        numeric_columns = window_data.select_dtypes(include=np.number)

        # Calcola i range interquartili per ciascuna feature nel sottogruppo di dati
        q1 = numeric_columns.quantile(0.15)
        q3 = numeric_columns.quantile(0.85)
        iqr = q3 - q1

        # Calcola i limiti inferiore e superiore per individuare gli outlier
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Identifica gli outlier per ciascuna feature e aggiungili alla lista degli outlier
        outlier_mask = ((numeric_columns < lower_bound) | (numeric_columns > upper_bound)).any(axis=1)
        outliers.extend(window_data[outlier_mask].index.tolist())

    return outliers

# Utilizzo dell'implementazione
outlier_indices = find_outliers_by_window(dataset)

'''appena fatto cio posso andare ad eliminare le righe che contengono gli outliers con tale codice'''


# Utilizza il metodo drop per eliminare le righe con gli indici specificati
df = dataset.drop(outlier_indices)
# Ora 'df' conterrà le righe originali meno quelle con gli indici specificati

# Salva il DataFrame in un file Excel
# Salva il DataFrame in un file Excel
df.to_excel(r"C:\Users\Utente\Desktop\Repository GitHub\tesi magistrale\data\features_segnali_controlli_100sigxpatient_nopatientwithlessthan100sig_nooutliers.xlsx", index=False)
# Imposta index=False se non vuoi includere gli indici nel file Excel