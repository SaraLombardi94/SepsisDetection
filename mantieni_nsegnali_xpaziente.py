# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 19:08:12 2024

@author: Utente
"""
import pandas as pd
import os

file_excel = r"C:\Users\Utente\Desktop\Repository GitHub\tesi magistrale\data\features_segnali_sepsi.xlsx"
OutDir = r"C:\Users\Utente\Desktop\Repository GitHub\tesi magistrale\data"

# Carico il file excel in un dataframe
df = pd.read_excel(file_excel)

# Estraggo i primi 7 caratteri dalla prima colonna
df["sottostringa"] = df[df.columns[0]].astype(str).str.slice(0, 7)

# Controllo per ogni riga se i primi 7 caratteri del valore sono uguali ai primi 7 caratteri del valore precedente
dup_mask = df["sottostringa"] == df["sottostringa"].shift()

# Applico un'altra maschera per identificare dove inizia una sequenza di più di 100 duplicati consecutivi
dup_mask = dup_mask.groupby((dup_mask != dup_mask.shift()).cumsum()).cumsum() > 100

# Applico la maschera al dataframe
df = df[~dup_mask]

# Elimino la colonna della sottostringa
df = df.drop(columns="sottostringa")

# Costruisco il percorso completo del file di output
output_file = os.path.join(OutDir, "features_segnali_sepsi_100sigxpatient.xlsx")

# Salvo il dataframe modificato in un nuovo file excel
df.to_excel(output_file, index=False)