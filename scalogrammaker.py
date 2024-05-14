# -*- coding: utf-8 -*-
"""
Created on Tue May 14 09:56:25 2024

@author: Utente
"""

import os 
import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
import tensorflow as tf
import ssqueezepy as sp
#from ssqueezepy.visuals import plot, imshow
from scipy.signal import cwt, morlet
from PIL import Image
import io



f = 125 #frequenza di campionamento
t_sample = 1/f #tempo di campionamento
n_samples = f*30 #numero di campioni considerato del segnale discreto


input_cartella = r'C:\Users\Utente\Desktop\minidataset\control'
output_cartella = r'C:\Users\Utente\Desktop\wetransfer_controls-microcirculation_2024-04-23_1250\controls-microcirculation\scalogrammi\control'

# Definizione delle scale per la CWT
#custom_scales = np.arange(41, 125)#○,3159,dtype='float64')  # Gli intervalli di scale possono essere adattati a seconda delle necessità



'''funzione che rimuove la componente continua a 0Hz del segnale'''
def sottrai_media(lista):
    mean_value = np.mean(lista)
    nuova_lista = np.array([valore - mean_value for valore in lista])
    return nuova_lista



# Itera attraverso i percorsi dei file NPZ
for nome_file_npz in os.listdir(input_cartella):
    if 'npz' in nome_file_npz:
        
        
        
        
        percorso_file_npz = os.path.join(input_cartella, nome_file_npz)
        # Carica il file NPZ
        dati_npz = np.load(percorso_file_npz, allow_pickle=True)
        dati_npz.files
        sig = dati_npz['arr_0'] #trasformo il segnale in una lista di valori
        sig = sig[:n_samples]
        
        Wx, scales, c = sp.cwt(sig, 'morlet', scales='log', derivative=False,nv=32)
        Wx=np.abs(Wx)
        
        
        Wx_with_channels= Wx[:, :, np.newaxis]
        Wx_rgb = np.concatenate([Wx_with_channels]*3, axis=-1)
        
        
        plt.title('Scalogramma')
        plt.imshow(image)
        plt.show()
        
        # # Salva il plot in un buffer utilizzando BytesIO
        # buf = io.BytesIO()
        # plt.savefig(buf, format='png',transparent=False)
        # plt.close()
        # buf.seek(0)  # Muovi il puntatore all'inizio del file
      
        # # Usa Pillow per aprire l'immagine dal buffer
        # image = Image.open(buf)
        
        # # Forza la conversione a RGB per rimuovere il canale alpha
        # rgb_image = image.convert('RGB')
        
        # # Converte l'immagine in un array NumPy
        # image_array = np.array(rgb_image)
        
        
        # #pulisci il buffer se non lo usi piu
        # buf.close()
        
        #fare operazioni con image_array
        
        
        '''
        ###### prima di fare la trasformata, elimino la componente continua ######
        sig_dyn = sottrai_media(sig)
        #### trasformata di Fourier ######
        sig_ft = fftpack.fft(sig_dyn)
        Amplitude = np.abs(sig_ft) #trovo l'ampiezza della trasformata in ogni punto
        Amplitude = np.log(Amplitude + 1)  # Aggiungiamo 1 per evitare il logaritmo di zero
        sample_freq = fftpack.fftfreq(sig.size, d=t_sample) # per trovare la frequenza ad ogni campione, devo passare all'interno della funzione fftfreq sia la dimensione del segnale che il tempo di campionamento del segnale stesso
        # Visualizzazione del risultato
        plt.figure(figsize=(10, 4))
        plt.plot(sample_freq, Amplitude)
        plt.title('FFT del Segnale')
        plt.xlabel('Frequenza (Hz)')
        plt.ylabel('Log Magnitude')
        plt.grid(True)
        plt.show()
        '''
        
        
        