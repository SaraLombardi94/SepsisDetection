# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 12:35:36 2024

@author: Sara
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statannot import add_stat_annotation
from sklearn.preprocessing import RobustScaler
from scipy.stats import mannwhitneyu
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm

# Carica i dati
file_path = r'D:/phD_Sara/microcircolo/Sepsis/datasets/controls-microcirculation/datasetSeed4_features/allfeaturesForStatisticalAnalysis.xlsx'
df = pd.read_excel(file_path)
df = df.rename(columns={'class': 'group_class'})
cols_to_scale = [col for col in df.columns if col not in ['subject_id', 'group_class']]

# Inizializza lo standard scaler
scaler = RobustScaler()
# Standardizza le colonne
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

# Assicurati che la variabile 'group_class' sia categorica
df['group_class'] = df['group_class'].astype(str)

agg_results = []

for col in cols_to_scale:
    print(f"\n Analyzing {col} parameter \n")
    print(f"LINEAR MIXED MODEL\n")
    # Definisci il modello misto lineare
    model = mixedlm(f"{col} ~ group_class", df, groups=df["subject_id"], re_formula="~group_class")
    # Fitta il modello
    try:
        result = model.fit()
        # Mostra i risultati
        print(result.summary())
    except Exception as e:
        print(f"Unable to fit model for {col}: {e}")
    
    # Aggrega le misure ripetute (ad esempio, prendendo la mediana per ogni soggetto)
    agg_df = df.groupby('subject_id').agg({f'{col}': 'median', 'group_class': 'first'}).reset_index()
    agg_df['Parametro'] = col
    agg_results.append(agg_df)
    
    # Dividi i gruppi
    group0 = agg_df[agg_df['group_class'] == '0'][col]
    group1 = agg_df[agg_df['group_class'] == '1'][col]
    print(f"MANN-WHITNEYU")
    # Esegui il test di Mann-Whitney U
    stat, p_value = mannwhitneyu(group0, group1)
    
    # Mostra i risultati
    print('Statistic:', stat)
    print('p-value:', p_value)

# Concatenate tutti i risultati aggregati
agg_all = pd.concat(agg_results)
agg_all.to_excel(r"C:\Users\Sara\Desktop\tesiMartina\file_for_boxplot.xlsx",index=False)

# Numero di parametri per grafico
params_per_plot = 6

# Creazione dei boxplot per gruppi di parametri
for i in range(0, len(cols_to_scale), params_per_plot):
    subset_params = cols_to_scale[i:i + params_per_plot]
    plt.figure(figsize=(20, 10))
    subset_data = agg_all[agg_all['Parametro'].isin(subset_params)]
    melted_data = pd.melt(subset_data, id_vars=['subject_id', 'group_class', 'Parametro'], value_vars=subset_params, var_name='Parameter', value_name='Value')
    melted_data = melted_data.dropna()
    melted_data = melted_data.drop(columns='Parametro')
    
    ax = sns.boxplot(x='Parameter', y='Value', hue='group_class', data=melted_data, palette="Set2")
    plt.title(f'Boxplot dei parametri da {i+1} a {i+len(subset_params)} per classi')
    plt.xlabel('Parametro')
    plt.ylabel('Valore')

    # Aggiungi annotazioni di significatività statistica per ogni parametro
    for param in subset_params:
        add_stat_annotation(ax, data=melted_data[melted_data['Parameter'] == param], x='Parameter', y='Value',
                            hue='group_class', box_pairs=[("0", "1")],
                            test='Mann-Whitney', text_format='star', loc='inside', verbose=2)

    plt.legend(title='Classe')
    plt.show()
