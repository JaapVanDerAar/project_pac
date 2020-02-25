#%% imports

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%% Change directory to import necessary modules

os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab\Code\distinguish_pac')

#%% import self-defined modules

import module_load_data as load_data
import module_detect_pac as detect_pac
from module_load_data import get_signal

from bycycle.filt import lowpass_filter
from bycycle.features import compute_features

#%% Load human data

# metadata
subjects = ['al','ca','cc','de','fp','gc','gf',
          'h0','hh','jc','jm','jp','mv','rh','rr',
          'ug','wc','wm','zt']
os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab')
dat_name = 'fixation_pwrlaw'
fs = 1000

# timewindow
timewindow = 100000/fs # 110800 is the length of shortest recording

# load data function for human   
datastruct_human, _ = load_data.load_data_human(dat_name, subjects, fs, timewindow)

#%% create features dataframes

# human dataframe
num_epochs_human = 1
size_human = sum([len(datastruct_human[subj]) for subj in range(len(datastruct_human))])

features_df_human = pd.DataFrame(index=range(0,size_human*num_epochs_human), columns=['subj', 'day', 'ch', 'ep'])

# fill in dataframe with days always 0 
features_df_human['subj'] = [subj for subj in range(len(datastruct_human)) 
                        for ch in range(len(datastruct_human[subj]))
                        for ep in range(num_epochs_human)]
features_df_human['day'] = 0
features_df_human['ch'] = [ch for subj in range(len(datastruct_human)) 
                        for ch in range(len(datastruct_human[subj]))
                        for ep in range(num_epochs_human)]
features_df_human['ep'] = [ep for subj in range(len(datastruct_human)) 
                        for ch in range(len(datastruct_human[subj]))
                        for ep in range(num_epochs_human)]

data_dict = {'human': datastruct_human}
features_df = {'human': features_df_human}

#%% FOOOF

freq_range = [4, 48] # for peak detection
bw_lims = [2, 6]
max_n_peaks = 5

pd.options.mode.chained_assignment = None

for key in data_dict: 


    # initialize space in features_df     
    # periodic component
    features_df[key]['CF'] = np.nan
    features_df[key]['Amp'] = np.nan
    features_df[key]['BW'] = np.nan
    
    # aperiodic component
    features_df[key]['offset'] = np.nan
    features_df[key]['knee'] = np.nan
    features_df[key]['exp'] = np.nan
    
    for ii in range(416,448):
      
        
        signal, fs = get_signal(data_dict, features_df, key, ii)
        
    
        if ~np.isnan(signal).any(): 
            
            
            # compute frequency spectrum
            freq_mean, psd_mean = spectral.compute_spectrum(signal, fs, method='welch', avg_type='mean', nperseg=fs*2)
            
            # if dead channel
            if sum(psd_mean) == 0: 
                
                # no oscillation was found
                features_df[key]['CF'][ii] = np.nan
                features_df[key]['Amp'][ii] = np.nan
                features_df[key]['BW'][ii] = np.nan
                
            else:
                
                # Initialize FOOOF model
                fm = FOOOF(peak_width_limits=bw_lims, background_mode='knee', max_n_peaks=max_n_peaks)
                
                # fit model
                fm.fit(freq_mean, psd_mean, freq_range) 
    
                fm.report()
                
                # Central frequency, Amplitude, Bandwidth
                peak_params = fm.peak_params_
                
                #offset, knee, slope
                background_params = fm.background_params_
                                
                    
                # if peaks are found
                if len(peak_params) > 0: 
                    
                    # find which peak has the biggest amplitude
                    max_ampl_idx = np.argmax(peak_params[:,1])
                    
                    # find this peak hase the following characteristics:
                    # 1) CF between 4 and 12 Hz
                    # 2) Amp above .2
                    if ((peak_params[max_ampl_idx][0] < 24) &  \
                        (peak_params[max_ampl_idx][0] > 4) &  \
                        (peak_params[max_ampl_idx][1] >.15)):
                        
                        # write oscillation parameters to dataframe
                        features_df[key]['CF'][ii] = peak_params[max_ampl_idx][0]
                        features_df[key]['Amp'][ii] = peak_params[max_ampl_idx][1]
                        features_df[key]['BW'][ii] = peak_params[max_ampl_idx][2]
                                     
                    # otherwise write empty
                    else:   
                        features_df[key]['CF'][ii] = np.nan
                        features_df[key]['Amp'][ii] = np.nan
                        features_df[key]['BW'][ii] = np.nan
                        
                # if no peaks are found, write empty
                elif len(peak_params) == 0:
                    
                    # write empty
                    features_df[key]['CF'][ii] = np.nan
                    features_df[key]['Amp'][ii] = np.nan
                    features_df[key]['BW'][ii] = np.nan
                
                # add backgr parameters to dataframe
                features_df[key]['offset'][ii] = background_params[0]
                features_df[key]['knee'][ii] = background_params[1]
                features_df[key]['exp'][ii] = background_params[2]
                
                
                print('this was ch', ii)
