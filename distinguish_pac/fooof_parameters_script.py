import numpy as np
import os
import pandas as pd

os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab\Code\distinguish_pac')

from neurodsp import spectral
from fooof import FOOOF

#%% Load data 

# change dir 
os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab')

# data
datastruct = np.load('datastruct_rats.npy', allow_pickle=True)

# dataframe
features_df = pd.read_csv('features_df_rats.csv', sep=',')


#%% 

# original is [4, 55], [2, 8], 4
freq_range = [4, 48] # for peak detection
bw_lims = [2, 6]
max_n_peaks = 5
# Add absolute threshold 'min_peak_height
# freq_range_long = [4, 118] # to establish a more reliable slope
fs = 1500



for ii in range(0,10):
  
    # get data
    subj = features_df['subj'][ii]
    day = features_df['day'][ii]
    ch = features_df['ch'][ii]
    ep = features_df['ep'][ii]

    sig = datastruct[subj][day][ch][ep]
    
    # compute frequency spectrum
    freq_mean, psd_mean = spectral.compute_spectrum(sig, fs, method='welch', avg_type='mean', nperseg=fs*2)
    
    # if dead channel
    if sum(psd_mean) == 0: 
        
        # no oscillation was found
        features_df['CF'][ii] = np.nan
        features_df['Amp'][ii] = np.nan
        features_df['BW'][ii] = np.nan
        
    else:
        
        # Initialize FOOOF model
        fm = FOOOF(peak_width_limits=bw_lims, background_mode='knee', max_n_peaks=max_n_peaks)
        
        # fit model
        fm.fit(freq_mean, psd_mean, freq_range) 

        fm.report()