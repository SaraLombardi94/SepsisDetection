# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 17:23:47 2024

@author: paolo
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, mannwhitneyu

#step 1: load the dataset
df = pd.read_excel(r'G:/Repository GitHub/tesi magistrale/data/all_features_with_subid.xlsx')

# Step 2: Grouping Data
grouped_data = df.groupby('label')

# Step 3: Feature Analysis
for feature in df.columns:
    if feature != 'SubId' and feature != 'label' and feature != 'nome_segnale':  # Exclude non-feature columns
        if pd.api.types.is_numeric_dtype(df[feature]):  # Check if the column is numeric
            print("Feature:", feature)
            print(grouped_data[feature].describe())
            print()

# Step 4: Visualization
for feature in df.columns:
    if feature != 'SubId' and feature != 'label' and feature != 'nome_segnale':  # Exclude non-feature columns
        if pd.api.types.is_numeric_dtype(df[feature]):  # Check if the column is numeric
            g = sns.FacetGrid(df, col="label", height=6)
            g.map(sns.distplot, feature, kde=True)
            plt.suptitle(f'Distribution of {feature}')
            plt.show()

# Step 5: Statistical Tests (t-test and Mann-Whitney U test)
for feature in df.columns:
    if feature != 'SubId' and feature != 'label' and feature != 'nome_segnale':  # Exclude non-feature columns
        if pd.api.types.is_numeric_dtype(df[feature]):  # Check if the column is numeric
            control_data = df[df['label'] == 0][feature]
            sick_data = df[df['label'] == 1][feature]

            # Performing Mann-Whitney U test
            u_statistic, p_value_u = mannwhitneyu(control_data, sick_data)

            print("Feature:", feature)
            print("Mann-Whitney U test:")
            print("   U-statistic:", u_statistic)
            print("   p-value:", p_value_u)
            if p_value_u < 0.05:
                print("   Significant difference between groups (Mann-Whitney U test)")
            else:
                print("   No significant difference between groups (Mann-Whitney U test)")

            print()
            
            
# Step 6: Create a new DataFrame with significant features
significant_features = pd.DataFrame()

for feature in df.columns:
    if feature != 'SubId' and feature != 'label' and feature != 'nome_segnale':  # Exclude non-feature columns
        if pd.api.types.is_numeric_dtype(df[feature]):  # Check if the column is numeric
            control_data = df[df['label'] == 0][feature]
            sick_data = df[df['label'] == 1][feature]

            # Performing Mann-Whitney U test
            u_statistic, p_value_u = mannwhitneyu(control_data, sick_data)

            if p_value_u < 0.05:  # Check if either test shows significance
                significant_features[feature] = df[feature]

# Add additional columns to the new DataFrame
significant_features['nome_segnale'] = df['nome_segnale']
significant_features['label'] = df['label']
significant_features['SubId'] = df['SubId']

#step 8: Remove the column R2_ of fit perche tale feature mi da info solo sul fitting del modello, ma niente riguardo al segnale
significant_features.drop(columns=['R2_of_fit'], inplace=True)



# Step 7: Save the new DataFrame to an Excel file
#significant_features.to_excel("significant_features.xlsx", index=False)

