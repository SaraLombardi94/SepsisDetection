# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 10:49:53 2024

@author: Utente
"""

import csv
import random
import pandas as pd
from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=25)
#model = svm.SVC()
#model = Perceptron()



# Carica il file XLSX
df = pd.read_excel(r"D:/phD_Sara/tesiPaolo/SepsisDetection/data/input machine learning_nomediane.xlsx")

# Seleziona tutte le colonne tranne la prima
df_selected = df.iloc[:, 1:]

# Salva il DataFrame selezionato in un file CSV
df_selected.to_csv("file.csv", index=False)

# Read data in from file
with open("file.csv") as f:
    reader = csv.reader(f)
    next(reader)

    data = []
    for row in reader:
        data.append({
            "evidence": [float(cell) for cell in row[:8]],
            "label": "sano" if row[8] == "0" else "patologico"
        })

# Separate data into training and testing groups
holdout = int(0.40 * len(data))
random.shuffle(data)
testing = data[:holdout]
training = data[holdout:]

# Train model on training set
X_training = [row["evidence"] for row in training]
y_training = [row["label"] for row in training]
model.fit(X_training, y_training)

# Make predictions on the testing set
X_testing = [row["evidence"] for row in testing]
y_testing = [row["label"] for row in testing]
predictions = model.predict(X_testing)

# Compute how well we performed
correct = 0
incorrect = 0
total = 0
for actual, predicted in zip(y_testing, predictions):
    total += 1
    if actual == predicted:
        correct += 1
    else:
        incorrect += 1

# Print results
print(f"Results for model {type(model).__name__}")
print(f"Correct: {correct}")
print(f"Incorrect: {incorrect}")
print(f"Accuracy: {100 * correct / total:.2f}%")