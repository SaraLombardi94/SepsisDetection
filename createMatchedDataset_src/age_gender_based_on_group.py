# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 15:22:06 2024
@author: Sara
"""

import pandas as pd
import os
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns

controlDemPath = r"D:/phD_Sara/data/healthyControls/clinicalData/controlData.xlsx"
comorbiditiesPath = r"D:/phD_Sara/data/control/clinicalData/demographicsControl.csv"
sepsisPath = r"D:/phD_Sara/data/sepsis/clinicalData/demographicsSepsis.csv"
outPath = r"D:\phD_Sara\microcircolo\Sepsis\data\clinicalData"


def replace_age(df):
    df['age'] = df['age'].replace(300, 90)
    return df

# Funzione per calcolare statistiche descrittive
def calculate_age_statistics(df):
    return df['age'].describe()

# Funzione per generare grafici della distribuzione di sesso ed età
def age_plot(df_list, group_names, colors):
    fig, axs = plt.subplots(3, 1, figsize=(10, 15), constrained_layout=True)
    fig.suptitle('Age Distribution Across Groups', fontsize=16)
    for ax, df, group, color in zip(axs, df_list, group_names, colors):
        sns.histplot(df['age'], bins=20, ax=ax, kde=False, color=color)
        ax.set_title(group, fontsize=14)
        ax.set_xlabel('Age', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.tick_params(labelsize=10)
    plt.show()

def plot_gender_distribution(dfs, group_names):
    # Creazione della figura e degli assi
    fig, axs = plt.subplots(len(dfs), 1, figsize=(8, 10), constrained_layout=True)
    # Ciclo per ogni gruppo di dati e relativo subplot
    for i, (df, name) in enumerate(zip(dfs, group_names)):
        # Conteggio del genere e creazione del grafico a barre
        gender_count = df['gender'].value_counts().reindex(['M', 'F'])  # Assicurarsi che l'ordine sia Maschi e poi Femmine
        bars = sns.barplot(x=gender_count.index, y=gender_count.values, ax=axs[i], palette=["blue", "orange"])
        # Aggiunta del conteggio sopra le barre
        for bar in bars.patches:
            axs[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), int(bar.get_height()), 
                        ha='center', va='bottom', color='black', fontsize=10)
        # Impostazione del titolo e delle etichette
        axs[i].set_title(f'Gender Distribution in {name}', fontsize=12)
        axs[i].set_xlabel('Gender', fontsize=10)
        axs[i].set_ylabel('Count', fontsize=10)
        axs[i].tick_params(axis='x', labelsize=9)
        axs[i].tick_params(axis='y', labelsize=9)
    # Visualizzazione del grafico
    plt.show()

def calculate_statistics(dfs, group_names):
    stats_results = pd.DataFrame()
    for df, name in zip(dfs, group_names):
        # Calcolare le statistiche per l'età
        age_stats = df['age'].agg(['mean', 'median', 'std']).rename(f'{name}_age')
        stats_results = pd.concat([stats_results, age_stats], axis=1)
        
    return stats_results

    
controlData = pd.read_excel(controlDemPath)
# Adattamento del DataFrame control_data per uniformità
controlData.rename(columns={'Sesso': 'gender', 'Età': 'age'}, inplace=True)
comorbiditiesData = pd.read_csv(comorbiditiesPath)
sepsisData = pd.read_csv(sepsisPath)
comorbiditiesData = replace_age(comorbiditiesData)
sepsisData = replace_age(sepsisData)

datasets = [controlData, comorbiditiesData, sepsisData]
names = ['HealthyControl', 'Non-septic', 'Septic']
colors = ['skyblue', 'lightgreen', 'salmon']
# Creazione dei subplot
age_plot(datasets, names, colors)
plot_gender_distribution(datasets, names)
stats = calculate_statistics(datasets,names)
stats.to_excel(os.path.join(outPath,"statsAge.xlsx"),index=True)