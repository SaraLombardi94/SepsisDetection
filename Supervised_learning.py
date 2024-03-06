# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 15:52:18 2024

@author: Utente
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
from collections import Counter

# Carica il file XLSX
df = pd.read_excel(r"C:/Users/Utente/Desktop/Repository GitHub/tesi magistrale/data/all_features.xlsx")

# Estrai i primi 7 caratteri dalla colonna 'nome_segnale'
df['Patient_ID'] = df['nome_segnale'].str[:7]

# Riorganizza le colonne per mettere 'Patient_ID' come prima colonna
columns = list(df.columns)
columns.insert(0, columns.pop(columns.index('Patient_ID')))
df = df[columns]

# Rimuovi le righe con valori mancanti
df = df.dropna()

# Salva il DataFrame selezionato in un file CSV
df.to_csv("file.csv", index=False)

# Carica il dataset
dataset = pd.read_csv("file.csv")

# Raggruppa i dati per paziente
pazienti = dataset.groupby('Patient_ID')

# Inizializza i set di training e test
X_train, X_test, y_train, y_test = pd.DataFrame(), pd.DataFrame(), pd.Series(dtype='float64'), pd.Series(dtype='float64')

# Elenca tutti i pazienti
elenco_pazienti = list(pazienti.groups.keys())

# Numero di pazienti da includere nel set di test
num_pazienti_test = int(len(elenco_pazienti) * 0.2)  # Ad esempio, puoi impostare il 20% dei pazienti come set di test

# Scegli casualmente i pazienti da includere nel set di test
pazienti_test = np.random.choice(elenco_pazienti, size=num_pazienti_test, replace=False)

# Aggiungi i segnali di ogni paziente al set di training o test
for paziente, segnali in pazienti:
    X_paziente = segnali.drop(columns=['Patient_ID', 'nome_segnale', 'label'])  # Carica le features
    y_paziente = segnali['label']  # Carica le etichette
    
    if paziente in pazienti_test:
        # Aggiungi tutti i segnali del paziente al set di test
        X_test = pd.concat([X_test, X_paziente])
        y_test = pd.concat([y_test, y_paziente])
    else:
        # Aggiungi tutti i segnali del paziente al set di training
        X_train = pd.concat([X_train, X_paziente])
        y_train = pd.concat([y_train, y_paziente])

# Ora hai i set di training e test con tutti i segnali dei pazienti divisi correttamente.

# Inizializza il modello SVM
model = SVC(kernel='linear')

# Addestra il modello sui dati di training
model.fit(X_train, y_train)

# Effettua le predizioni sui dati di test
predictions = model.predict(X_test)

# Calcola l'accuratezza del modello sui dati di test
accuracy = accuracy_score(y_test, predictions)

# Stampa l'accuratezza
print("Accuracy:", accuracy)

# Inizializza una lista per memorizzare le accuratezze per ogni paziente
accuratezze_pazienti = []

# Per ogni paziente nel set di test
for paziente in pazienti_test:
    # Filtra solo i segnali del paziente corrente
    segnali_paziente = dataset[dataset['Patient_ID'] == paziente]
    
    # Prevedi l'etichetta del paziente basata sulla maggioranza delle predizioni dei segnali
    predizioni_paziente = model.predict(segnali_paziente.drop(columns=['Patient_ID', 'nome_segnale', 'label']))
    etichetta_predetta = Counter(predizioni_paziente).most_common(1)[0][0]
    
    # Ottieni l'etichetta reale del paziente
    etichetta_reale = segnali_paziente['label'].iloc[0]
    
    # Calcola l'accuratezza per il paziente corrente
    if etichetta_predetta == etichetta_reale:
        accuratezza_paziente = 1
    else:
        accuratezza_paziente = 0
    
    # Aggiungi l'accuratezza del paziente alla lista
    accuratezze_pazienti.append(accuratezza_paziente)

# Calcola l'accuratezza media su tutti i pazienti nel set di test
accuratezza_media = sum(accuratezze_pazienti) / len(accuratezze_pazienti)

# Stampa l'accuratezza media
print("Accuracy media per paziente:", accuratezza_media)


# Rimuovere la colonna 'nome_segnale' prima di calcolare le feature più importanti
features = dataset.drop(columns=['Patient_ID', 'nome_segnale', 'label'])

# Ottenere i vettori di supporto pesati
weights = model.coef_

# Moltiplicare i pesi per ottenere l'importanza relativa delle feature
feature_importance = np.abs(weights)

# Trovare l'indice delle feature più importanti
top_features_indices = np.argsort(feature_importance)[::-1]

# Stampare solo le prime 10 feature più importanti
print("Top 10 Feature Importances:")
for index in top_features_indices[0:10]:
    print(features.columns[index])



