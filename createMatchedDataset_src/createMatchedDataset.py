# -*- coding: utf-8 -*-
"""
Created on Thu Apr 4 14:01:24 2024
@author: Sara
"""
import os 
from glob import glob
import pandas as pd
from collections import defaultdict, Counter
import random
import numpy as np 
import shutil

datasetPath = "D:/phD_Sara/microcircolo/Sepsis/datasets/healthy-nonseptic-sepsis/matchedDataset.csv"
controlDir = r"D:\phD_Sara\data\healthyControls\TM_filtered\segments"
comorbDir = r"D:\phD_Sara\data\control\segments2min"
sepsisDir = r"D:\phD_Sara\data\sepsis\segments2min"
outDir = r"D:\phD_Sara\microcircolo\Sepsis\datasets\healthy-nonseptic-sepsis"
dirDataset = r"D:\phD_Sara\microcircolo\Sepsis\datasets\healthy-nonseptic-sepsis"

MAX_SAMPLES = 3 
random_seeds = [4,24,42,56,100,128,256,384,536,1024]
MOVE_FILE_IN_DATASET_DIR = True

def filter_paths(paths, subject_ids):
    # Crea un insieme per un accesso più rapido.
    subject_id_set = set(subject_ids)
    # Filtra i percorsi che contengono un subject_id valido.
    filtered_paths = [path for path in paths if any(sub_id in path for sub_id in subject_id_set)]
    return filtered_paths

# Funzione per aggiornare i conteggi basati sui percorsi forniti.
def count_subject_files(paths):
    for path in paths:
        # Estrai il subject_id dal nome del file. 
        # Assumiamo che il subject_id sia parte del nome del file prima di un "_" o qualsiasi altro separatore distintivo.
        subject_id = os.path.basename(path).split('-')[0]
        subject_counts[subject_id] += 1

# Funzione per selezionare al massimo 3 campioni per soggetto.
def select_samples(paths, max_samples):
    # Inizializza un dizionario per raccogliere i percorsi selezionati.
    selected_paths = defaultdict(list)
    
    # Organizza i percorsi per subject_id.
    for path in paths:
        # Estrai il subject_id dal nome del file.
        subject_id = os.path.basename(path).split('-')[0]
        # Aggiungi il percorso alla lista del soggetto.
        selected_paths[subject_id].append(path)
    
    # Seleziona al massimo 'max_samples' campioni per soggetto.
    for subject_id, paths in selected_paths.items():
        if len(paths) > max_samples:
            selected_paths[subject_id] = random.sample(paths, max_samples)
    
    return dict(selected_paths)

datasetInfo = pd.read_csv(datasetPath)
subject_ids = datasetInfo['subid'].astype(str).tolist()
pathControls = glob(os.path.join(controlDir,"*.npz"))
pathComorb = glob(os.path.join(comorbDir,"*.npz"))
pathSepsis = glob(os.path.join(sepsisDir,"*.npz"))
filtered_pathComorb = filter_paths(pathComorb, subject_ids)
filtered_pathSepsis = filter_paths(pathSepsis, subject_ids)
# Inizializza un dizionario per tenere traccia dei conteggi.
subject_counts = defaultdict(int)
# Conta i file per i soggetti comorbidi e per la sepsi.
count_subject_files(filtered_pathComorb)
count_subject_files(filtered_pathSepsis)
# Converti il dizionario di conteggio predefinito in un dizionario regolare per visualizzazione.
subject_counts_dict = dict(subject_counts)

for seed in random_seeds:
    random.seed(seed)
    selected_paths_comorb = select_samples(filtered_pathComorb,MAX_SAMPLES)
    selected_paths_sepsis = select_samples(filtered_pathSepsis, MAX_SAMPLES)
    print(f"Seed corrente: {seed}")
    print(f"Percorsi selezionati : {selected_paths_comorb.values()}")
 
    output_file_path = os.path.join(outDir,f"matched_dataset_seed{seed}.txt")

    # Scrivi tutti i percorsi nel file, senza considerare gli ID dei soggetti.
    with open(output_file_path, 'w') as file:
        for path in pathControls:
            file.write(path + '\n')
        for paths in selected_paths_comorb.values():
            for path in paths:
                file.write(path + '\n')
        for paths in selected_paths_sepsis.values():
            for path in paths:
                file.write(path + '\n')
                
if MOVE_FILE_IN_DATASET_DIR:
    datasets = glob(os.path.join(outDir,"*.txt"))
    for datasetPath in datasets:
        datasetName =  datasetPath.split(os.path.sep)[-1].rstrip(".txt")
        seedNumber = datasetName.split("_")[-1]
        os.makedirs(os.path.join(dirDataset,"control",seedNumber),exist_ok=True) 
        os.makedirs(os.path.join(dirDataset,"sepsis",seedNumber),exist_ok=True) 
        os.makedirs(os.path.join(dirDataset,"nonseptic",seedNumber),exist_ok=True) 
        fileList = np.loadtxt(datasetPath,dtype='str',comments=None)
        for filePath in fileList:
            if "healthyControls" in filePath:
                shutil.copy(filePath,os.path.join(dirDataset,"control",seedNumber))
            elif filePath.split(os.path.sep)[-3]=="control":
                shutil.copy(filePath,os.path.join(dirDataset,"nonseptic",seedNumber))
            elif "sepsis" in filePath:
                shutil.copy(filePath,os.path.join(dirDataset,"sepsis",seedNumber))
    