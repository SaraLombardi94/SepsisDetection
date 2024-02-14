# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:49:59 2024

@author: Utente
"""

import pandas as pd
import os

file_excel = r"C:/Users/Utente/Desktop/Repository GitHub/tesi magistrale/data/features_segnali_sepsi_100sigxpatient_nopatientwithlessthan100sig_nooutliers.xlsx"
OutDir = r"C:\Users\Utente\Desktop\Repository GitHub\tesi magistrale\data"

# Carica il file Excel
df = pd.read_excel(file_excel)

# Estrai i primi 7 caratteri dalla prima cella di ogni riga
def extract_patient_id(row):
    return row[0][:7]

# Crea una nuova colonna nel DataFrame per i primi 7 caratteri
df['Patient_ID'] = df.apply(extract_patient_id, axis=1)

# Inizializza un dizionario per raccogliere i dati
dati_risultati = {'ID paziente': [], 'Mediana della media': [], 'Mediana potenza totale': [], 'Mediana percentuale di potenza a VLF rispetto alla potenza totale': [],
                  'Mediana percentuale di potenza a LF rispetto alla potenza totale': [], 'Mediana percentuale di potenza a HF rispetto alla potenza totale': [], 'Mediana frequenza VLF dove ho il picco': [],
                  'Mediana frequenza LF dove ho il picco':[],'Mediana frequenza HF dove ho il picco':[]}

# Calcola la mediana delle features per ogni paziente
for paziente, group in df.groupby('Patient_ID'):
    # Estrai i primi 7 caratteri dalla colonna "nome_segnale"
    nome_segnale = group['nome_segnale'].str.slice(0, 7)
    dati_risultati['ID paziente'].append(nome_segnale.iloc[0])
    dati_risultati['Mediana della media'].append(group['media'].median())
    dati_risultati['Mediana potenza totale'].append(group['potenza totale'].median())
    dati_risultati['Mediana percentuale di potenza a VLF rispetto alla potenza totale'].append(group['percentuale di potenza a VLF rispetto alla potenza totale'].median())
    dati_risultati['Mediana percentuale di potenza a LF rispetto alla potenza totale'].append(group['percentuale di potenza a LF rispetto alla potenza totale'].median())
    dati_risultati['Mediana percentuale di potenza a HF rispetto alla potenza totale'].append(group['percentuale di potenza a HF rispetto alla potenza totale'].median())
    dati_risultati['Mediana frequenza VLF dove ho il picco'].append(group['frequenza VLF dove ho il picco'].median())
    dati_risultati['Mediana frequenza LF dove ho il picco'].append(group['frequenza LF dove ho il picco'].median())
    dati_risultati['Mediana frequenza HF dove ho il picco'].append(group['frequenza HF dove ho il picco'].median())


# Crea il DataFrame risultati
risultati = pd.DataFrame(dati_risultati)

# Salva il DataFrame in un nuovo file Excel
risultati.to_excel(os.path.join(OutDir, "risultati_mediana_per_paziente_sepsi.xlsx"), index=False)