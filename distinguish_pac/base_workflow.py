#%% Change directory and imports

# import standard libraries
import os
import numpy as np
import matplotlib.pyplot as plt



#%% Change directory to import necessary modules

os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab\Code\distinguish_pac')

#%% import self-defined modules

import module_load_data as load_data
import module_pac_functions as pacf
import module_detect_pac as detect_pac
import module_pac_plots as pac_plt

#%% Meta data

subjects = ['al','ca','cc','de','fp','gc','gf','gw',
          'h0','hh','jc','jm','jp','mv','rh','rr',
          'ug','wc','wm','zt']

os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab')

dat_name = 'fixation_pwrlaw'

#%% Parameters 

fs = 1000

timewindow = 5 # how many data want to select (in sec)

ss = 60 # where to start to select data per recording (in sec)

#%% Load data

datastruct, elec_locs = load_data.load_data_and_locs(dat_name, subjects, fs, timewindow, ss)

#%% Save datastructure

np.save('datastruct', datastruct)
np.save('elec_locs', elec_locs)

#%% Input parameters for bands

phase_providing_band = [4,8]; #4-8 Hz band
amplitude_providing_band = [80, 125]; #80-125 Hz band

#%% Detect PAC function

pac_presence, pac_pvals, pac_rhos = detect_pac.cal_pac_values(datastruct, phase_providing_band, amplitude_providing_band, fs)

#%% Resampled PAC

num_resamples = 1000

resamp_rho = detect_pac.resampled_pac(datastruct, phase_providing_band, amplitude_providing_band, fs, num_resamples)

#%% Save resampled data

os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab')

np.save('resamp_rho', resamp_rho)

#%% Calculate true pac values

pac_true_zvals, pac_true_pvals, pac_true_presence = detect_pac.true_pac_values(pac_rhos, resamp_rho)

# Which channels in which subjects have PAC?
pac_idx = list(np.where(pac_true_presence == 1))

#%% Save true values and pac index

os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab')
np.save('pac_true_zvals', pac_true_zvals)
np.save('pac_true_pvals', pac_true_pvals)
np.save('pac_true_presence', pac_true_presence)
np.save('pac_idx', pac_idx)

#%% 1 Signal plot

subj = 0 
ch = 0

fs = 1000
phase_providing_band = [4,8]; #4-8 Hz band
amplitude_providing_band = [80, 125]; #80-125 Hz band

pac_plt.plot_signal(datastruct, phase_providing_band, amplitude_providing_band, subj, ch, fs)

            
#%% PAC channels plot

fs = 1000
phase_providing_band = [4,8]; #4-8 Hz band
amplitude_providing_band = [80, 125]; #80-125 Hz band

for ii in range(len(pac_idx[0])):
    
    subj = pac_idx[0][ii]
    ch = pac_idx[1][ii]
    
    pac_plt.plot_signal(datastruct, phase_providing_band, amplitude_providing_band, subj, ch, fs)
    
#%% List of stuff to open

# standard modules
import os
import numpy as np

# change dir
os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab\Code\distinguish_pac')

# import self-defined modules
import module_load_data as load_data
import module_detect_pac as detect_pac
import module_pac_plots as pac_plt
   
# change dir 
os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab')

# data
datastruct = np.load('datastruct.npy', allow_pickle=True)
elec_locs = np.load('elec_locs.npy', allow_pickle=True)

subjects = ['al','ca','cc','de','fp','gc','gf','gw',
          'h0','hh','jc','jm','jp','mv','rh','rr',
          'ug','wc','wm','zt']

# resampled rho values
resamp_rho = np.load('resamp_rho.npy', allow_pickle=True) 

# resampled statistics    
pac_true_zvals = np.load('pac_true_zvals.npy')
pac_true_pvals = np.load('pac_true_pvals.npy')
pac_true_presence = np.load('pac_true_presence.npy')
pac_idx = np.load('pac_idx.npy')

# parameters
fs = 1000
phase_providing_band = [4,8]; #4-8 Hz band
amplitude_providing_band = [80, 125]; #80-125 Hz band

#%% Loop over all sign. PAC channels to assess them 

assess = []

for ii in range(len(pac_idx[0])):
    
    subj = pac_idx[0][ii]
    ch = pac_idx[1][ii]
    
    fig = plt.figure(ii)
    pac_plt.plot_signal(datastruct, phase_providing_band, amplitude_providing_band, subj, ch, fs)
    
    plt.draw()
    plt.pause(1) 
  
    while True:
        
        val = input("""Enter a number:
            1 = 'True' PAC
            2 = Sharp waveforms
            3 = Artifacts
            4 = Other 
            """)
            
        if  val == "1" or val == "2" or val == "3" or val == "4":
            
             assess.append(val)
             plt.close(fig)
             break        
    

