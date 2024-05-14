#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 09:31:52 2022

@author: saralombardi
"""
# get onset points informations 

import os
from glob import glob
import numpy as np
import scipy
from scipy import signal
import matplotlib.pyplot as plt
import sklearn 
from sklearn import preprocessing

#sampling Frequency
FS = 125 #Hz
 
#filter parameters
CUT_OFF_LOW = 0.5
CUT_OFF_HIGH = 5
ORDER = 1
PWD_MIN = round((60/180)*FS) #samples

#butterworth band pass filter 
def butter_bandpass(data, cutoff_low, cutoff_high, order):
    nyq = 0.5 * FS 
    normal_cutoff_low = cutoff_low / nyq
    normal_cutoff_high = cutoff_high / nyq
    b, a = signal.butter(order, [normal_cutoff_low, normal_cutoff_high], btype='bandpass', analog=False)
    filtered = signal.filtfilt(b, a, data)
    return filtered

def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
    """
    Input :
    s: 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
    split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
    Output :
    lmin,lmax : high/low envelope idx of input signal s
    """
    # locals min      
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1 
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1 

    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = np.mean(s) 
        # pre-sorting of locals min based on relative position with respect to s_mid 
        lmin = lmin[s[lmin]<s_mid]
        # pre-sorting of local max based on relative position with respect to s_mid 
        lmax = lmax[s[lmax]>s_mid]


    # global max of dmax-chunks of locals max 
    lmin = lmin[[i+np.argmin(s[lmin[i:i+dmin]]) for i in range(0,len(lmin),dmin)]]
    # global min of dmin-chunks of locals min 
    lmax = lmax[[i+np.argmax(s[lmax[i:i+dmax]]) for i in range(0,len(lmax),dmax)]]
    
    return lmin,lmax


def envelopePPG(sig, distance=PWD_MIN):
    # split signal into negative and positive parts
    u_x = np.where(sig > 0)[0]
    l_x = np.where(sig < 0)[0]
    u_y = sig.copy()
    u_y[l_x] = 0
    l_y = -sig.copy()
    l_y[u_x] = 0
    
    # find upper and lower peaks
    u_peaks, _ = scipy.signal.find_peaks(u_y, height=(0.15,2),distance=distance)
    l_peaks, _ = scipy.signal.find_peaks(l_y, height=(0.15,2), distance=distance)
    
    # use peaks and peak values to make envelope
    u_x = u_peaks
    u_y = sig[u_peaks]
    l_x = l_peaks
    l_y = sig[l_peaks]

    # plt.plot(sig,color='blue')
    # plt.scatter(u_x,u_y, color='r')
    # plt.scatter(l_x,l_y,color='g')
    # plt.show()
    return l_x, u_x

baseDir = r'C:\Users\Utente\Desktop\wetransfer_controls-microcirculation_2024-04-23_1250\controls-microcirculation\tf\microcirculation_seed4'
outputDir = r'C:\Users\Utente\Desktop\wetransfer_controls-microcirculation_2024-04-23_1250\controls-microcirculation\tf\microcirculation_seed4'
#samples = glob(os.path.join(sampleDir,'*.npz'))

seeds = ['seed4', 'seed24', 'seed42', 'seed56', 'seed100', 'seed128', 'seed256', 'seed384', 'seed536', 'seed1024']


# Cicla su tutte le cartelle seed
#for seed in seeds:
#    sampleDir = os.path.join(baseDir, seed)
#    outputDir = sampleDir  # Salva l'output nella stessa cartella di input
samples = glob(os.path.join(baseDir, '*.npz'))
for samplePath in samples:
    filename = samplePath.split(os.path.sep)[-1].removesuffix('.npz')
    file = np.load(samplePath)['arr_0'].flatten()
    file = sklearn.preprocessing.minmax_scale(file, feature_range = (-1,1))
    filtered = butter_bandpass(file,CUT_OFF_LOW,CUT_OFF_HIGH,ORDER)
    #lmin, lmax = hl_envelopes_idx(filtered,dmin=1,dmax=1,split=True)
    lmin, lmax = envelopePPG(filtered)
    # plt.plot(file, color='blue', label='original')
    # plt.plot(lmin, file[lmin], 'r', label='low')
    # plt.plot(lmax, file[lmax], 'g', label='high')
    # plt.legend()
    # plt.show()
    np.savetxt(os.path.join(outputDir,f'{filename}.txt'),lmin)   

    

               