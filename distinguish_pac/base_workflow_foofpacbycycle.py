#%% Import standard libraries
import os
import numpy as np


#%% Import Voytek Lab libraries

from neurodsp.utils import create_times
from neurodsp.plts.spectral import plot_power_spectra
from neurodsp.plts.time_series import plot_time_series
from neurodsp import spectral
from neurodsp import filt

from fooof import FOOOF

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


# initialze storing array
psd_peaks = []

for subj in range(len(datastruct)):
    
    # initialize channel specific storage array
    psd_peak_chs = []
    
    for ch in range(len(datastruct[subj])):
        
        # get signal
        sig = datastruct[subj][ch]
        
        # compute frequency spectrum
        freq_mean, psd_mean = spectral.compute_spectrum(sig, fs, method='welch', avg_type='mean', nperseg=fs*2)
     
        # Set the frequency range upon which to fit FOOOF
        freq_range = [4, 55]
        bw_lims = [2, 8]
        max_n_peaks = 4
        
        if sum(psd_mean) == 0: 
            
            peak_params = np.empty([0, 3])
            
            psd_peak_chs.append(peak_params)
            
        else:
            
            # Initialize FOOOF model
            fm = FOOOF(peak_width_limits=bw_lims, background_mode='knee', max_n_peaks=max_n_peaks)
            
            fm.fit(freq_mean, psd_mean, freq_range) 
            
            # offset, knee, slope
            background_params = fm.background_params_
            
            # Central frequency, Amplitude, Bandwidth
            peak_params = fm.peak_params_
            
            if len(peak_params) > 0: 
                
                # find which peak has the biggest amplitude
                max_ampl_idx = np.argmax(peak_params[:,1])
                    
                # define biggest peak in power spectrum and add to channel array
                psd_peak_chs.append(peak_params[max_ampl_idx])
            
            elif len(peak_params) == 0:
                
                psd_peak_chs.append(peak_params)
                
    psd_peaks.append(psd_peak_chs)
        

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
