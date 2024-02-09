# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 16:21:17 2024

@author: Utente
"""

# Importo le librerie necessarie 
import pandas as pd 
import matplotlib.pyplot as plt 


file_excel =r"C:\Users\Utente\Desktop\Repository GitHub\tesi magistrale\data\features_segnali_controlli_100sigxpatient.xlsx"
# Carico i dati delle features da un file Excel 
# Assumo che il file abbia una colonna per ogni feature e una riga per ogni segnale 
# Assumo che il file si chiami "features.xlsx" e sia nella stessa cartella del codice 
data = pd.read_excel(file_excel).iloc[102:200]  # Considero solo le prime 100 righe
'''
# Creo una figura con quattro sottografici 
figura, grafici = plt.subplots(2, 2) 

# Creo un istogramma della feature "Importo" nel primo sottografico 
grafici[0, 0].hist(data["media"], bins=10, color="orange", edgecolor="grey") 
grafici[0, 0].set_xlabel("Importo") 
grafici[0, 0].set_ylabel("Frequenza") 
grafici[0, 0].set_title("Istogramma di Importo") 

# Creo un grafico a linee della feature "Importo" nel secondo sottografico 
grafici[0, 1].plot(data["media"], color="blue", marker="o") 
grafici[0, 1].set_xlabel("Indice") 
grafici[0, 1].set_ylabel("Importo") 
grafici[0, 1].set_title("Grafico a linee di Importo") 

# Creo un grafico a barre della feature "Importo" nel terzo sottografico 
grafici[1, 0].bar(data.index, data["media"], color="green", width=0.8) 
grafici[1, 0].set_xlabel("Indice") 
grafici[1, 0].set_ylabel("Importo") 
grafici[1, 0].set_title("Grafico a barre di Importo") 

# Creo un grafico a torta della feature "Importo" nel quarto sottografico 
grafici[1, 1].pie(data["media"], labels=data.index, autopct="%1.1f%%") 
grafici[1, 1].set_title("Grafico a torta di Importo") 
'''

# Creo un istogramma della feature "media"
plt.hist(data["potenza totale"], bins=10, color="orange", edgecolor="grey") 
plt.xlabel("potenza totale") 
plt.ylabel("Frequenza") 
plt.title("Istogramma di potenza totale") 
plt.show()

# Mostro la figura con tutti i grafici 
plt.show()
