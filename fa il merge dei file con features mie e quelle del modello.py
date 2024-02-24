# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 11:43:56 2024

@author: Utente
"""
import pandas as pd

# Leggi i due file Excel
df1 = pd.read_excel(r'C:/Users/Utente/Desktop/Repository GitHub/tesi magistrale/data/expModeling/medie_modello_merged.xlsx')
df2 = pd.read_excel(r'C:/Users/Utente/Desktop/Repository GitHub/tesi magistrale/data/featuresPaolo/features_paolo_merged.xlsx')


# Trova l'intersezione dei pazienti presenti nei due DataFrame
pazienti_comuni = set(df1['nome_segnale']).intersection(set(df2['nome_segnale']))

# Filtra i dati solo per i pazienti comuni
df1_comuni = df1[df1['nome_segnale'].isin(pazienti_comuni)]
df2_comuni = df2[df2['nome_segnale'].isin(pazienti_comuni)]

# Unisci i due DataFrame utilizzando la colonna paziente come chiave di unione
df_completo = pd.merge(df1_comuni, df2_comuni, on='nome_segnale')


# Salva il DataFrame combinato in un nuovo file Excel
df_completo.to_excel(r'C:\Users\Utente\Desktop\Repository GitHub\tesi magistrale\data\all_features.xlsx', index=False)
