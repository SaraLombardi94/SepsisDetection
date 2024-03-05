# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 10:43:08 2024

@author: paolo
"""

import sys
sys.path.append(r'C:\Users\paolo\OneDrive\Desktop')
from Mann_Whitney_test import significant_features
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

# Assicurati che significant_features contenga solo le colonne delle features e 'label'
# Rimuovi 'SubId' se non è necessaria per l'addestramento
X = significant_features.drop(columns=['label', 'nome_segnale', 'SubId'])
y = significant_features['label']
# Sostituisci i valori mancanti con la media della colonna
X.fillna(X.mean(), inplace=True)
groups = significant_features['SubId']  # Colonna SubId per i gruppi

# Definizione dei modelli
models = {
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC(probability=True),
    'Gradient Boosting': GradientBoostingClassifier()
}

# Definizione della strategia di validazione con tenuta utilizzando GroupKFold
# Imposta il numero di fold desiderato (ad esempio, 5)
num_folds = 5
group_kfold = GroupKFold(n_splits=num_folds)

# Inizializzazione delle liste per memorizzare i risultati delle metriche
metrics = {
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1-score': [],
    'AUC': []
}

# Ciclo su ogni modello
for model_name, model in models.items():
    print(f"Training {model_name}")
    # Inizializzazione delle liste per memorizzare i risultati delle metriche per il modello corrente
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    auc_scores = []

    # Ciclo su ogni fold per la validazione con tenuta
    for train_index, test_index in group_kfold.split(X, y, groups):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Addestramento del modello
        model.fit(X_train, y_train)

        # Valutazione del modello
        y_pred = model.predict(X_test)

        # Calcolo delle metriche di performance
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        auc_scores.append(auc)

    # Calcolo della media delle metriche di performance per il modello corrente
    mean_accuracy = np.mean(accuracy_scores)
    mean_precision = np.mean(precision_scores)
    mean_recall = np.mean(recall_scores)
    mean_f1 = np.mean(f1_scores)
    mean_auc = np.mean(auc_scores)

    # Salvataggio delle metriche di performance medie per il modello corrente
    metrics['Accuracy'].append(mean_accuracy)
    metrics['Precision'].append(mean_precision)
    metrics['Recall'].append(mean_recall)
    metrics['F1-score'].append(mean_f1)
    metrics['AUC'].append(mean_auc)

    # Stampa delle metriche di performance medie per il modello corrente
    print(f"Mean Accuracy ({model_name}):", mean_accuracy)
    print(f"Mean Precision ({model_name}):", mean_precision)
    print(f"Mean Recall ({model_name}):", mean_recall)
    print(f"Mean F1-score ({model_name}):", mean_f1)
    print(f"Mean AUC ({model_name}):", mean_auc)

# Stampa delle metriche di performance medie su tutti i modelli
for metric_name, metric_values in metrics.items():
    mean_metric = np.mean(metric_values)
    print(f"Mean {metric_name} (All Models):", mean_metric)
