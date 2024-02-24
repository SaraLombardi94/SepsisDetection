# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 18:11:32 2024

@author: Utente
"""

import pandas as pd
import os

#contatore
i = 0
# 1. Leggi il file Excel
df_excel = pd.read_excel(r'C:/Users/Utente/Desktop/Repository GitHub/tesi magistrale/data/featuresPaolo/control/features_segnali_controlli.xlsx')

# 2. Ottieni la lista dei file nella cartella di destinazione
cartella_destinazione = r'C:\Users\Utente\Desktop\Repository GitHub\tesi magistrale\data\expModeling\control'
elenco_file = os.listdir(cartella_destinazione)

# 3. Estrai i nomi dei file dalla colonna desiderata del file Excel
nomi_da_trovare = df_excel['nome_segnale'].tolist()

# 4. Filtra i file nella cartella in base ai nomi estratti
file_selezionati = [file for file in elenco_file if any(nome in file for nome in nomi_da_trovare)]


empty_df = pd.DataFrame()

# 5. Esegui le operazioni desiderate sui file selezionati
for file in file_selezionati:
    # Esegui le operazioni desiderate su ogni file
    # Ad esempio, puoi aprire il file e fare operazioni su di esso
    file_path = os.path.join(cartella_destinazione, file)
    with open(file_path, 'r') as f:
        df = pd.read_csv(f)
        empty_df.at[i, 'nome_segnale'] = file #aggiungo il nome del segnale 
        # Filtra le righe con R2_of_fit inferiore a 0.90
        df_filtrato = df[df['R2_of_fit'] >= 0.90]
        # Calcola la media di ciascuna caratteristica
        medie_caratteristiche = df_filtrato.mean()
        # Aggiungi le medie come nuove righe al DataFrame empty_df
        for caratteristica, media in medie_caratteristiche.items():
            empty_df.at[i, caratteristica] = media
        i = i+1
    
#rimuovo dalla colonna nome_segnale la parte .npz.csv
empty_df['nome_segnale'] = empty_df['nome_segnale'].str.replace('.npz.csv', '')

# Salva il DataFrame delle medie in un nuovo file Excel
empty_df.to_excel(r'C:/Users/Utente/Desktop/Repository GitHub/tesi magistrale/data/expModeling/medie_modello_controlli.xlsx', index=False)


