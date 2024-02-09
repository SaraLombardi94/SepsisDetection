# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 17:36:40 2024

@author: Utente
"""

import pandas as pd


file_excel = r"C:\Users\Utente\Desktop\Repository GitHub\tesi magistrale\data\features_segnali_sepsi_100sigxpatient_nopatientwithlessthan100sig.xlsx"
OutDir = r"C:\Users\Utente\Desktop\Repository GitHub\tesi magistrale\data"



# Carica il file Excel
df = pd.read_excel(file_excel)

# Calcola il numero di pazienti
num_pazienti = len(df) // 100

# Crea una lista di pazienti per ogni riga
pazienti = [i // 100 + 1 for i in range(len(df))]

# Aggiungi la colonna 'Paziente' al DataFrame
df['Paziente'] = pazienti

# Inizializza un dizionario per raccogliere i dati
dati_risultati = {'nome_segnale': [], 'Media della media': [], 'Media potenza totale': [], 'Media percentuale di potenza a VLF rispetto alla potenza totale': []}

# Calcola la media delle features per ogni paziente
for paziente, group in df.groupby('Paziente'):
    dati_risultati['nome_segnale'].append(df.loc[df['Paziente'] == paziente, 'nome_segnale'].iloc[0])
    dati_risultati['Media della media'].append(group['media'].mean())
    dati_risultati['Media potenza totale'].append(group['potenza totale'].mean())
    dati_risultati['Media percentuale di potenza a VLF rispetto alla potenza totale'].append(group['percentuale di potenza a VLF rispetto alla potenza totale'].mean())

# Crea il DataFrame risultati
risultati = pd.DataFrame(dati_risultati)

# Salva il DataFrame in un nuovo file Excel
risultati.to_excel(OutDir, "risultati_media_per_paziente.xlsx", index=False)
