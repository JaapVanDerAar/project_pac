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
#import module_detect_pac as detect_pac
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


#%% Filter out 60 Hz and cut off beginning and end of data GIVES PROBLEMS

## Filter frequency range
#f_range = (58, 62)
#
#for subj in range(len(datastruct)):
#    for ch in range(len(datastruct[subj])):
#        
#        datastruct[subj][ch] = filt.filter_signal(datastruct[subj][ch], fs, 'bandstop', f_range, n_seconds=0.5)
#        
#        # cut off beginning and end of data (1s) which are nan due to filter
#        datastruct[subj][ch] = datastruct[subj][ch][1000:len(datastruct[subj][ch])-1000]
#%% Save structure

np.save('datastruct_fpb', datastruct)

#%% Calculate PSD Welch method

subj = 0
ch = 0 

sig = datastruct[subj][ch]

# Plot the loaded signal
times = create_times(len(sig)/fs, fs)
plot_time_series(times, sig, xlim=[0, 3])

freq_mean, psd_mean = spectral.compute_spectrum(sig, fs, method='welch', avg_type='mean', nperseg=fs*2)

plot_power_spectra(freq_mean[:200], psd_mean[:200], 'Welch')

# Set the frequency range upon which to fit FOOOF
freq_range = [2, 50]
bw_lims = [2, 8]
max_n_peaks = 4


# Initialize FOOOF model
fm = FOOOF(peak_width_limits=bw_lims, background_mode='knee', max_n_peaks=max_n_peaks)

fm.fit(freq_mean, psd_mean, freq_range) 

# Run FOOOF model - calculates model, plots, and prints results
fm.report(freq_mean, psd_mean, freq_range)

# offset, knee, slope
background_params = fm.background_params_

# Central frequency, Amplitude, Bandwidth
peak_params = fm.peak_params_

# find which peak has the biggest amplitude
max_ampl_idx = np.argmax(peak_params[:,1])

# define biggest peak in power spectrum
psd_peak = peak_params[max_ampl_idx]


#%% 



