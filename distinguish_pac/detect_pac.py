#%% import packages

import os
import scipy.io
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, filtfilt, hilbert
import pickle

#%% Set database

os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab')

# dataset
dataset = 'fixation_pwrlaw'
fs = 1000

#%% subject list

subjects=['al','ca','cc','de','fp','gc','gf','gw',
          'h0','hh','jc','jm','jp','mv','rh','rr',
          'ug','wc','wm','zt']


#%% Filtering and circle correlation functions

def butter_bandpass(lowcut, highcut, fs, order=4):
    #lowcut is the lower bound of the frequency that we want to isolate
    #hicut is the upper bound of the frequency that we want to isolate
    #fs is the sampling rate of our data
    nyq = 0.5 * fs #nyquist frequency - see http://www.dspguide.com/ if you want more info
    low = float(lowcut) / nyq
    high = float(highcut) / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(mydata, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, mydata)
    return y

def circCorr(ang,line):
    n = len(ang)
    rxs = sp.stats.pearsonr(line,np.sin(ang))
    rxs = rxs[0]
    rxc = sp.stats.pearsonr(line,np.cos(ang))
    rxc = rxc[0]
    rcs = sp.stats.pearsonr(np.sin(ang),np.cos(ang))
    rcs = rcs[0]
    rho = np.sqrt((rxc**2 + rxs**2 - 2*rxc*rxs*rcs)/(1-rcs**2)) #r
    r_2 = rho**2 #r squared
    pval = 1- sp.stats.chi2.cdf(n*(rho**2),1)
    standard_error = np.sqrt((1-r_2)/(n-2))

    return rho, pval, r_2,standard_error

#%% Selecting frequency bands 
    
phase_providing_band = [4,8]; #4-8 Hz band
amplitude_providing_band = [80, 125]; #80-125 Hz band


#%% Loop through every subj and channel to find which have PAC

# create output matrix of 20 * 64 (subj * channels)
PAC_presence = np.full((20,64),np.nan)

# for every subject
for subj in range(len(subjects)): 
    
    # get the filename
    sub_label = subjects[subj] + '_base'
    filename = os.path.join(os.getcwd(), dataset, 'data', sub_label)
    
    # load data
    dataStruct = sp.io.loadmat(filename)
    data = dataStruct['data']
    locs = dataStruct['locs']
    
    # for every channel 
    for ch in range(len(locs)):
        
        
    
        #calculating phase of theta
        phase_data = butter_bandpass_filter(data[:,ch], phase_providing_band[0], phase_providing_band[1], round(float(fs)));
        phase_data_hilbert = hilbert(phase_data);
        phase_data_angle = np.angle(phase_data_hilbert);
        
        #calculating amplitude envelope of high gamma
        amp_data = butter_bandpass_filter(data[:,ch], amplitude_providing_band[0], amplitude_providing_band[1], round(float(fs)));
        amp_data_hilbert = hilbert(amp_data);
        amp_data_abs = abs(amp_data_hilbert);


        PAC_values = circCorr(phase_data_angle, amp_data_abs)
        
        if PAC_values[1] <= 0.05:
            
            PAC_presence[subj, ch] = 1
            
        elif PAC_values[1] > 0.05: 

            PAC_presence[subj, ch] = 0
            
            
    print('another one is done =), this was subj', subj)
            
            
    np.save(PAC_presence, PAC_presence)      
            
     
#%% pickle dump and load 
       
# dump information to that file
pickle.dump(PAC_presence, open('PAC_presence', 'wb'))

PAC_pres = pickle.load(open("PAC_presence", "rb"))
        
        
#%%   

(PAC_presence == 1).sum()
    
    
#%% Plot some data


subj = 8
ch = 8

# get the filename
sub_label = subjects[subj] + '_base'
filename = os.path.join(os.getcwd(), dataset, 'data', sub_label)

# load data
dataStruct = sp.io.loadmat(filename)
data = dataStruct['data']
locs = dataStruct['locs']


#calculating phase of theta
phase_data = butter_bandpass_filter(data[:,ch], phase_providing_band[0], phase_providing_band[1], round(float(fs)));
phase_data_hilbert = hilbert(phase_data);
phase_data_angle = np.angle(phase_data_hilbert);

#calculating amplitude envelope of high gamma
amp_data = butter_bandpass_filter(data[:,ch], amplitude_providing_band[0], amplitude_providing_band[1], round(float(fs)));
amp_data_hilbert = hilbert(amp_data);
amp_data_abs = abs(amp_data_hilbert);


PAC_values = circCorr(phase_data_angle, amp_data_abs)


signal_both_bands = phase_data_hilbert + amp_data_hilbert


plt.figure(figsize = (20,8));
plt.plot((signal_both_bands[1:int(fs)]),label= 'Combined')
plt.plot((amp_data_hilbert[1:int(fs)]),label= 'High Gamma')
plt.plot((phase_data_hilbert[1:int(fs)]),label= 'Theta')



#%% 

freqs, psd = signal.welch(data[:,ch])

plt.figure(figsize=(5, 4))
plt.semilogx(freqs, psd)
plt.title('PSD: power spectral density')
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.tight_layout()
