#%% imports 

import numpy as np
import os
import pandas as pd

os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab\Code\distinguish_pac')
from bycycle.filt import lowpass_filter
from bycycle.features import compute_features
from bycycle.burst import plot_burst_detect_params

#%% Load data 

# change dir 
os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab')

# data
datastruct = np.load('datastruct.npy', allow_pickle=True)

# dataframe
features_df = pd.read_csv('features_df.csv', sep=',')

#%% 




#%% parameters

fs = 1000
Fs = 1000
# epoch length
epoch_len_seconds = 20 # in seconds

f_lowpass = 35
N_seconds = 2

burst_kwargs = {'amplitude_fraction_threshold': 0.25,
                'amplitude_consistency_threshold': .4,
                'period_consistency_threshold': .45,
                'monotonicity_threshold': .6,
                'N_cycles_min': 3}


burst_kwargs = {'amplitude_fraction_threshold': .3,
                'amplitude_consistency_threshold': .4,
                'period_consistency_threshold': .5,
                'monotonicity_threshold': .8,
                'N_cycles_min': 5}

#%%


# for every channel with pac
for ii in range(10,20):
    
    # for every channel that has peaks
    if ~np.isnan(features_df['CF'][ii]):
    
        # define phase providing band
        CF = features_df['CF'][ii]
        BW = features_df['BW'][ii]
        
        print(CF)
        print(BW)
        
        phase_providing_band= [(CF - (BW/2)),  (CF + (BW/2))]
               
        subj = features_df['subj'][ii]
        ch = features_df['ch'][ii]
        ep = features_df['ep'][ii]
        data = datastruct[subj][ch][ep]
        
        signal = lowpass_filter(data, fs, f_lowpass, N_seconds=N_seconds, remove_edge_artifacts=False)
        
        bycycle_df = compute_features(signal, fs, phase_providing_band, burst_detection_kwargs=burst_kwargs)

#        plot_burst_detect_params(signal, Fs, bycycle_df,
#                         burst_kwargs, tlims=(0, 5), figsize=(16, 3), plot_only_result=True)
#
#        plot_burst_detect_params(signal, Fs, bycycle_df,
#                         burst_kwargs, tlims=(0, 5), figsize=(16, 3))
        
        # find biggest length with no violations 
        
        # save this part to a dataframe that has length[ii] of features_df
        
        # check whether these times are consistent with the timewindow of signal
        
        
        



#%% 

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from bycycle.filt import lowpass_filter


signal = data
Fs = 1000
phase_providing_band = [(CF - (BW/2)),  (CF + (BW/2))]
f_lowpass = 30
N_seconds = .2

signal_low = lowpass_filter(signal, Fs, f_lowpass,
                            N_seconds=N_seconds[jj], remove_edge_artifacts=False,
                            plot_frequency_response=True)
plt.plot(signal_low[0:2000])







# Plot frequency response of bandpass filter
from bycycle.filt import bandpass_filter
test = lowpass_filter(signal, Fs, phase_providing_band, N_seconds=.2, plot_frequency_response=True)
plt.plot(test[0:2000])

# Plot signal
t = np.arange(0, len(signal)/Fs, 1/Fs)
tlim = (2, 5)
tidx = np.logical_and(t>=tlim[0], t<tlim[1])

plt.figure(figsize=(12, 2))
plt.plot(t[tidx], signal[tidx], '.5')
plt.plot(t[tidx], signal_low[tidx], 'k')
plt.xlim(tlim)



#%% 
from bycycle.filt import bandpass_filter



bandpass_filter(signal, Fs, (4, 10), N_seconds=.75, plot_frequency_response=True)
N_seconds = [2]
  

for jj in range(len(N_seconds)):
    
  # for every channel with pac
    for ii in range(10,15):
        
        # for every channel that has peaks
        if ~np.isnan(features_df['CF'][ii]):
        
            # define phase providing band
            CF = features_df['CF'][ii]
            BW = features_df['BW'][ii]
            
            print(CF)
            print(BW)
            
            phase_providing_band= [(CF - (BW/2)),  (CF + (BW/2))]
                   
            subj = features_df['subj'][ii]
            ch = features_df['ch'][ii]
            ep = features_df['ep'][ii]
            data = datastruct[subj][ch][ep]
            
            bandpass_filter(data, Fs, phase_providing_band, N_seconds=N_seconds[jj], plot_frequency_response=True)
    














