# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 16:20:19 2024

@author: Utente
"""
import os 
import numpy as np


def check_duration(directory, sampling_rate=125, min_duration=31):
    """
    Controlla la durata dei file npz in una cartella specificata.

    Parameters:
    directory (str): Percorso della cartella contenente i file npz.
    sampling_rate (int): Frequenza di campionamento dei dati nei file npz.
    min_duration (int): Durata minima in secondi sotto la quale il nome del file viene stampato.
    """
    # Calcola il numero minimo di campioni per la durata minima specificata
    min_samples = sampling_rate * min_duration
    
    # Elenco tutti i file nella directory specificata
    for filename in os.listdir(directory):
        if filename.endswith('.npz'):
            filepath = os.path.join(directory, filename)
            # Carica il file npz
            data = np.load(filepath)
            # Assume che i dati siano salvati con una chiave che possiamo chiamare 'arr_0'
            if 'arr_0' in data:
                signal = data['arr_0']
                # Controlla la lunghezza del segnale
                if signal.size < min_samples:
                    
                    print(f"Il file {filename} ha una durata inferiore a {min_duration} secondi.")
                    #os.remove(filepath)
            #else:
                #print(f"Nessun dato trovato in {filename} con chiave 'arr_0'.")
            data.close()

# Usa la funzione con il percorso della cartella desiderata
check_duration(r'C:\Users\Utente\Desktop\wetransfer_controls-microcirculation_2024-04-23_1250\controls-microcirculation\tf\microcirculation_seed4')
