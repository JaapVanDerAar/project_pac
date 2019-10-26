#%% Import standard libraries
import os
import numpy as np


#%% Import Voytek Lab libraries



#%% Change directory to import necessary modules

os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab\Code\distinguish_pac')

#%% import self-defined modules

import module_load_data as load_data
#import module_pac_functions as pacf
import module_detect_pac as detect_pac
#import module_pac_plots as pac_plt

#%% Meta data

subjects = ['al','ca','cc','de','fp','gc','gf','gw',
          'h0','hh','jc','jm','jp','mv','rh','rr',
          'ug','wc','wm','zt']

os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab')

dat_name = 'fixation_pwrlaw'

#%% Parameters 

fs = 1000

timewindow = 110000/fs # 110800 is the length of shortest recording

#%% Run load data middle timewindow function
   
datastruct, elec_locs = load_data.load_data_timewindow(dat_name, subjects, fs, timewindow)


#%% Save structure

# np.save('datastruct_fpb', datastruct)

#%% Calculate largest peak in PSD using FOOF

psd_peaks = detect_pac.fooof_highest_peak(datastruct, fs)


#%% So now we have the peaks. Use those as input phase frequency for detecting PAC
    
amplitude_providing_band = [80, 125]; #80-125 Hz band

pac_presence, pac_pvals, pac_rhos = detect_pac.cal_pac_values_varphase(datastruct, amplitude_providing_band, fs, psd_peaks)

#%% How many channels have PAC when channels are NOT resampled?

array_size = np.size(pac_presence)
nan_count = np.isnan(pac_presence).sum().sum()
pp = (pac_presence == 1).sum().sum()
percentage_pac = pp / (array_size - nan_count) * 100
print('in ', percentage_pac, 'channels is PAC (unresampled)')


#%% Run resampling with variable phase and for full time frame of data

num_resamples = 1000

resamp_rho_varphase = detect_pac.resampled_pac_varphase(datastruct, amplitude_providing_band, fs, num_resamples, psd_peaks)

#%% Save resampled data

os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab')

np.save('resamp_rho_varphase', resamp_rho_varphase)

#%% Calculate true PAC values (by comparing rho value to resampled rho values, 
#   and assuming normal distribution, which seems like that)

