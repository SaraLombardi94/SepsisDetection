# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 15:52:18 2024

@author: Utente
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Carica il file XLSX
df = pd.read_excel(r"C:/Users/Utente/Desktop/Repository GitHub/tesi magistrale/data/features_segnali_merged_nooutliers_100sigxpatient_nolessthan100_prova.xlsx")


# Salva il DataFrame selezionato in un file CSV
df.to_csv("file.csv", index=False)

# Carica il dataset
dataset = pd.read_csv("file.csv")

# Raggruppa i dati per paziente
pazienti = dataset.groupby('Patient_ID')

# Inizializza i set di training e test
X_train, X_test, y_train, y_test = pd.DataFrame(), pd.DataFrame(), pd.Series(dtype='float64'), pd.Series(dtype='float64')

# Per ciascun paziente, aggiungi i segnali al set di training o test
for paziente, segnali in pazienti:
    X_paziente = segnali.drop(columns=['Patient_ID','nome_segnale', 'label'])  # Carica le features
    y_paziente = segnali['label']  # Carica le etichette

    # Dividi i segnali del paziente in set di training e test
    X_train_paziente, X_test_paziente, y_train_paziente, y_test_paziente = train_test_split(X_paziente, y_paziente, test_size=0.2, random_state=42)

    # Aggiungi i segnali di training e test ai set globali
    X_train = pd.concat([X_train, X_train_paziente])
    X_test = pd.concat([X_test, X_test_paziente])
    y_train = pd.concat([y_train, y_train_paziente])
    y_test = pd.concat([y_test, y_test_paziente])

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