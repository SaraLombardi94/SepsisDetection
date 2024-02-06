# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 15:50:58 2024

@author: Utente
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack
from scipy.integrate import simps

f = 125 #frequenza di campionamento
t_sample = 1/f #tempo di campionamento
n_samples = f*120 #numero di campioni considerato del segnale discreto

# Definisci il percorso della cartella contenente i file
cartella = r"D:\phD_Sara\data\control\segments2min"
outDir = r"D:\phD_Sara\tesiPaolo"


'''array con i tempi'''
t = []
for i in range(n_samples):
    t.append(t_sample*i) #in questo modo ho creato un array con tutti gli istanti temporali
i=0 #contatore che mi serve poi per aggiungere le features al file excel

'''funzioni che servono per trovare features nel dominio del tempo'''
def find_index_local_peaks(signal):
    indici_massimi = []
    for i in range(len(signal)):
        # Definisci l'intervallo intorno a ciascun elemento
        start = max(0, i - 80)
        end = min(len(signal), i + 81)
        # Verifica se l'elemento corrente è un massimo locale
        if signal[i] == max(signal[start:end]):
            indici_massimi.append(i)
    return indici_massimi
def rimuovi_duplicati_consecutivi(signal):
    new_indexes = []
    for valore in signal:
        # Aggiungi il valore solo se è diverso dall'ultimo valore aggiunto
        if not new_indexes or new_indexes[-1] != valore:
            new_indexes.append(valore)
    return new_indexes
def rimuovi_indici_max_adiacenti(lista):
    new_lista = []
    for i in range(0,len(lista)-1):
        if lista[i]-1 != lista[i-1]:
            new_lista.append(lista[i])
    return new_lista
def find_peaks_time_interval(list_of_peaks_indexes):
    intervals = []
    intervals.append(list_of_peaks_indexes[0]*t_sample)
    for i in range(len(list_of_peaks_indexes)-1):
        interval = (list_of_peaks_indexes[i+1]-list_of_peaks_indexes[i])*t_sample
        intervals.append(interval)
    return intervals
'''funzione che rimuove la componente continua a 0Hz del segnale'''
def sottrai_media(lista):
    mean_value = np.mean(lista)
    nuova_lista = np.array([valore - mean_value for valore in lista])
    return nuova_lista
'''funzione che rimuove le frequenze negative'''
def rimuovi_colonne_negative(array):
    # Estrai la seconda riga dell'array
    seconda_riga = array[1, :]
    # Trova gli indici delle colonne con valori maggiori di 0.04 nella seconda riga
    indici_da_rimuovere = np.where(seconda_riga < 0.00)[0]
    # Rimuovi le colonne dall'array
    nuovo_array = np.delete(array, indici_da_rimuovere, axis=1)
    return nuovo_array
'''funzione che mantiene il segnale a bassissima frequenza (0,00-0,04 Hz)'''
def rimuovi_colonne(array):
    # Estrai la seconda riga dell'array
    seconda_riga = array[1, :]
    
    # Trova gli indici delle colonne con valori maggiori di 0.04 nella seconda riga
    indici_da_rimuovere = np.where(seconda_riga > 0.04)[0]
    
    # Rimuovi le colonne dall'array
    nuovo_array = np.delete(array, indici_da_rimuovere, axis=1)
    
    return nuovo_array
'''funzione che mantiene il segnale a bassa frequenza 0.04Hz < Freq < 0.15 Hz '''
def rimuovi_colonne_LF(array):
    # Estrai la seconda riga dell'array
    seconda_riga = array[1, :]
    
    # Trova gli indici delle colonne con valori maggiori di 0.04 nella seconda riga
    indici_da_rimuovere = np.where((seconda_riga < 0.04) | (seconda_riga > 0.15))[0]
    
    # Rimuovi le colonne dall'array
    nuovo_array = np.delete(array, indici_da_rimuovere, axis=1)
    
    return nuovo_array
'''funzione che mantiene il segnale ad alta frequenza 0.15 Hz < Freq < 0.4 Hz '''
def rimuovi_colonne_HF(array):
    # Estrai la seconda riga dell'array
    seconda_riga = array[1, :]
    
    # Trova gli indici delle colonne con valori maggiori di 0.04 nella seconda riga
    indici_da_rimuovere = np.where((seconda_riga < 0.15) | (seconda_riga > 0.4))[0]
    
    # Rimuovi le colonne dall'array
    nuovo_array = np.delete(array, indici_da_rimuovere, axis=1)
    
    return nuovo_array
'''funzione che prende in ingresso le frequenze vlf, la psd a vlf, le frequenze totali e la psd totale e restituisce la percentuale di potenza del segnale in vlf, la potenza del segnale nelle frequenze vlf e la potenza totale'''
def calcola_percentuale_vlf_psd(frequenze_vlf, psd_vlf, frequenze_tot, psd_tot):
    # Calcola la potenza totale del segnale
    potenza_totale = simps(psd_tot, frequenze_tot)

    # Calcola la potenza nel VLF
    potenza_vlf = simps(psd_vlf, frequenze_vlf)

    # Calcola la percentuale di potenza nel VLF rispetto alla potenza totale
    percentuale_vlf = (potenza_vlf / potenza_totale) * 100

    return percentuale_vlf, potenza_vlf, potenza_totale
'''funzione che prende in ingresso le frequenze lf, la psd a lf, le frequenze totali e la psd totale e restituisce la percentuale di potenza del segnale in lf e la potenza del segnale nelle frequenze lf'''
def calcola_percentuale_lf_psd(frequenze_lf, psd_lf, frequenze_tot, psd_tot):
    # Calcola la potenza totale del segnale
    potenza_totale = simps(psd_tot, frequenze_tot)

    # Calcola la potenza nel VLF
    potenza_lf = simps(psd_lf, frequenze_lf)

    # Calcola la percentuale di potenza nel VLF rispetto alla potenza totale
    percentuale_lf = (potenza_lf / potenza_totale) * 100

    return percentuale_lf, potenza_lf
'''funzione che prende in ingresso le frequenze hf, la psd a hf, le frequenze totali e la psd totale e restituisce la percentuale di potenza del segnale in hf e la potenza del segnale nelle frequenze hf'''
def calcola_percentuale_hf_psd(frequenze_hf, psd_hf, frequenze_tot, psd_tot):
    # Calcola la potenza totale del segnale
    potenza_totale = simps(psd_tot, frequenze_tot)

    # Calcola la potenza nel VLF
    potenza_hf = simps(psd_hf, frequenze_hf)

    # Calcola la percentuale di potenza nel VLF rispetto alla potenza totale
    percentuale_hf = (potenza_hf / potenza_totale) * 100

    return percentuale_hf, potenza_hf

'''creo un dizionario vuoto dove aggiungo poi le varie features'''
features = {}


'''leggo il file excel che poi devo riempire con le features'''
df = pd.DataFrame()


'''posso creare una lista con tutti i file npz nella cartella oppure no'''
# Ottieni la lista di tutti i file NPZ nella cartella
#file_npz_nella_cartella = [file for file in os.listdir(cartella) if file.endswith('.npz')]

# Itera attraverso i percorsi dei file NPZ
for nome_file_npz in os.listdir(cartella):
    percorso_file_npz = os.path.join(cartella, nome_file_npz)
    #features["NomeFile"].append(nome_file_npz)
    #features["par1"].append(i)
    # Carica il file NPZ
    dati_npz = np.load(percorso_file_npz)
    dati_npz.files
    sig_totale = dati_npz['arr_0'] #trasformo il segnale in una lista di valori
    sig = sig_totale[:n_samples] #del segnale prendo solo i campioni che voglio io definiti all'interno della variabile n_samples
    
    '''trovo le features nel dominio del tempo'''
    mean = np.mean(sig) #valor medio del segnale
    std = np.std(sig) #deviazione standard del segnale
    
    '''trovo i massimi locali del segnale'''
    indici_max_start = find_index_local_peaks(sig)
    indici_max = rimuovi_indici_max_adiacenti(indici_max_start)
    peaks = sig[indici_max]
    '''trovo l'intervallo temporale tra i vari massimi locali'''
    peaks_intervals = find_peaks_time_interval(indici_max)
    '''trovo la media temporale dei picchi'''
    HR_medio = np.mean(peaks_intervals)

    '''prima di fare la trasformata, elimino la componente continua'''
    sig_nodc = sottrai_media(sig)
    '''trasformata di Fourier'''
    sig_ft = fftpack.fft(sig_nodc)
    Amplitude = np.abs(sig_ft) #trovo l'ampiezza della trasformata in ogni punto
    Power = Amplitude**2 #trovo la potenza della trasformata in ogni punto
    Angle = np.angle(sig_ft) #trovo l'angolo della trasformata in ogni punto 
    sample_freq = fftpack.fftfreq(sig.size, d=t_sample) # per trovare la frequenza ad ogni campione, devo passare all'interno della funzione fftfreq sia la dimensione del segnale che il tempo di campionamento del segnale stesso
    Amp_Freq = np.array([Amplitude, sample_freq]) #faccio un array con le varie ampiezze e le varie frequenze a tali ampiezze
    Amp_Freq_positive = rimuovi_colonne_negative(Amp_Freq)
    psd_posit = Amp_Freq_positive[0]**2 #psd del segnale senza frequenze negative
    '''trovo le features in frequenza'''
    amp_freq_VLF = rimuovi_colonne(Amp_Freq_positive) #mantengo il segnale a VLF
    Amp_position_VLF = amp_freq_VLF[0,:].argmax() #trovo la posizione in frequenza del picco di ampiezza massimo
    peak_freq_VLF = amp_freq_VLF[1, Amp_position_VLF] #trovo la frequenza di tale picco in ampiezza
    peak_amp_VLF = amp_freq_VLF[0, Amp_position_VLF]
    #trovo la potenza del segnale nelle VLF e la % di tale potenza rispetto alla potenza totale
    VLF_perc, potenza_VLF,potenza_totale = calcola_percentuale_vlf_psd(amp_freq_VLF[1], amp_freq_VLF[0]**2,Amp_Freq_positive[1],psd_posit)
   
    amp_freq_LF = rimuovi_colonne_LF(Amp_Freq_positive)
    Amp_position_LF = amp_freq_LF[0,:].argmax() #trovo la posizione in frequenza del picco di ampiezza massimo
    peak_freq_LF = amp_freq_LF[1, Amp_position_LF] #trovo la frequenza di tale picco in ampiezza
    peak_amp_LF = amp_freq_LF[0, Amp_position_LF]
    #trovo la potenza del segnale nelle LF e la % di tale potenza rispetto alla potenza totale
    LF_perc, potenza_LF = calcola_percentuale_lf_psd(amp_freq_LF[1], amp_freq_LF[0]**2,Amp_Freq_positive[1],psd_posit)
   
    amp_freq_HF = rimuovi_colonne_HF(Amp_Freq_positive)
    Amp_position_HF = amp_freq_HF[0,:].argmax() #trovo la posizione in frequenza del picco di ampiezza massimo
    peak_freq_HF = amp_freq_HF[1, Amp_position_HF] #trovo la frequenza di tale picco in ampiezza
    peak_amp_HF = amp_freq_HF[0, Amp_position_HF]
    #trovo la potenza del segnale nelle HF e la % di tale potenza rispetto alla potenza totale
    HF_perc, potenza_HF = calcola_percentuale_hf_psd(amp_freq_HF[1], amp_freq_HF[0]**2,Amp_Freq_positive[1],psd_posit)
    
    '''aggiungo le varie features al file excel'''
    df.at[i, 'nome_segnale'] = nome_file_npz
    df.at[i, 'media'] = mean
    df.at[i, 'std'] = std
    df.at[i, 'Heart rate medio'] = HR_medio
    df.at[i, 'potenza totale'] = potenza_totale
    df.at[i, 'potenza VLF'] = potenza_VLF
    df.at[i, 'potenza LF'] = potenza_LF
    df.at[i, 'potenza HF'] = potenza_HF
    df.at[i, 'percentuale di potenza a VLF rispetto alla potenza totale'] = VLF_perc
    df.at[i, 'percentuale di potenza a LF rispetto alla potenza totale'] = LF_perc
    df.at[i, 'percentuale di potenza a HF rispetto alla potenza totale'] = HF_perc
    df.at[i, 'frequenza VLF dove ho il picco'] = peak_freq_VLF
    df.at[i, 'Ampiezza del picco in VLF'] = peak_amp_VLF
    df.at[i, 'frequenza LF dove ho il picco'] = peak_freq_LF
    df.at[i, 'Ampiezza del picco in LF'] = peak_amp_LF
    df.at[i, 'frequenza HF dove ho il picco'] = peak_freq_HF
    df.at[i, 'Ampiezza del picco in HF'] = peak_amp_HF
    #faccio un contatore i che permette di aggiungere le features al file excel in maniera corretta  
    i= i+1
    # Chiudi il file NPZ per liberare la memoria
    dati_npz.close()
    
# Fine del ciclo, tutti i file NPZ sono stati elaborati    
# Salva il DataFrame aggiornato nel file Excel
df.to_excel(os.path.join(outDir,'features_segnali_controlli.xlsx'), index=False)
    
      
    
    
    
    












