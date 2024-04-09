# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 11:33:47 2024
@author: Sara
"""

import pandas as pd
import os
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np

controlDemPath = r"D:/phD_Sara/data/healthyControls/clinicalData/controlData.xlsx"
comorbiditiesPath = r"D:/phD_Sara/data/control/clinicalData/demographicsControl.csv"
sepsisPath = r"D:/phD_Sara/data/sepsis/clinicalData/demographicsSepsis.csv"
outPath = r"D:\phD_Sara\microcircolo\Sepsis\datasets\healthy-nonseptic-sepsis"

comorbiditiesDataDir = r"D:\phD_Sara\data\control\segments2min"
sepsisDataDir = r"D:\phD_Sara\data\sepsis\segments2min"

def extractSubjectsID(datadir):
    idlist = []
    for path in glob(os.path.join(datadir,"*.npz")):
        filename = path.split(os.path.sep)[-1]
        subid = filename.split('-')[0]
        subid = subid[2:]
        idlist.append(subid)
    return np.unique(np.array(idlist,dtype='int64'))

def generate_class_age(df):
    bins = [0,30,40,50,60,90]
    labels = ["18-29", "30-39","40-49", "50-59", "60+"]
    df['age'] = df['age'].replace(300, 90) # remove over 90 subjects identified as 300 in mimic database
    age_category = pd.cut(df['age'], bins=bins, labels=labels, include_lowest=True)
    df['age_category'] = age_category
    return df

def add_age_gender_category(dataframe):
    bins = [0,30,40,50,60,90]
    labels = ["18-29", "30-39","40-49", "50-59", "60+"]
    dataframe['age'] = dataframe['age'].replace(300, 90)
    # Crea la colonna 'age_category'
    dataframe['age_category'] = pd.cut(dataframe['age'], bins=bins, labels=labels, include_lowest=True)
    # Crea una colonna combinata che include sia l'età che il genere
    dataframe['age_gender_category'] = dataframe['age_category'].astype(str) + "_" + dataframe['gender']
    return dataframe

controlData = pd.read_excel(controlDemPath)[["Posizione Pleth", "Età", "Sesso"]]
# Adattamento del DataFrame control_data per uniformità
controlData.rename(columns={"Posizione Pleth": "subject_id","Sesso": "gender", "Età": "age"}, inplace=True)
comorbiditiesData = pd.read_csv(comorbiditiesPath)[["subject_id", "age", "gender"]]
sepsisData = pd.read_csv(sepsisPath)[["subject_id", "age", "gender"]]
controlData = add_age_gender_category(controlData)
comorbiditiesData = add_age_gender_category(comorbiditiesData)
sepsisData = add_age_gender_category(sepsisData)

# Utilizza Counter per ottenere il conteggio
number_of_subjects = Counter(controlData["age_gender_category"])
outDict = {"class":[], "subid":[], "age":[], "gender":[]}

comorbSubData = extractSubjectsID(comorbiditiesDataDir)
sepsisSubData = extractSubjectsID(sepsisDataDir)
# Stampa i conteggi per ogni categoria
for category, count in number_of_subjects.items():
    control_category = controlData[controlData["age_gender_category"]==category].reset_index(drop=True)
    sepsis_category = sepsisData[sepsisData["age_gender_category"]==category].reset_index(drop=True)
    sepsis_category = sepsis_category[sepsis_category['subject_id'].isin(sepsisSubData)]
    comorb_category = comorbiditiesData[comorbiditiesData["age_gender_category"]==category].reset_index(drop=True)
    comorb_category = comorb_category[comorb_category['subject_id'].isin(comorbSubData)]
    if len(sepsis_category)>count:
        sepsis_category = sepsis_category[:count]
    else:
        print(category,len(sepsis_category),count)
    if len(comorb_category)>count:
        comorb_category = comorb_category[:count]
    else:
        print(category,len(comorb_category),count)
    for i in range(count):
        outDict['class'].append("healthyControls")
        outDict['subid'].append(f"pleth{control_category.iloc[i]['subject_id']}")
        outDict['age'].append(control_category.iloc[i]['age'])
        outDict['gender'].append(control_category.iloc[i]['gender'])
    
    for i in range(len(comorb_category)):
        outDict['class'].append("non septic")
        outDict['subid'].append(f"p0{comorb_category.iloc[i]['subject_id']}")
        outDict['age'].append(comorb_category.iloc[i]['age'])
        outDict['gender'].append(comorb_category.iloc[i]['gender'])
        
    for i in range(len(sepsis_category)):
        outDict['class'].append("sepsis")
        outDict['subid'].append(f"p0{sepsis_category.iloc[i]['subject_id']}")
        outDict['age'].append(sepsis_category.iloc[i]['age'])
        outDict['gender'].append(sepsis_category.iloc[i]['gender'])

outDataset = pd.DataFrame(outDict)
outDataset.to_csv(os.path.join(outPath,"matchedDataset.csv"),index=False)