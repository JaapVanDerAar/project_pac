import os
import numpy as np
import scipy.io as sio


import glob
import h5py
import pandas as pd
import matplotlib.pyplot as plt
#%% Change directory to import necessary modules

os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab\Code\distinguish_pac')

#%% import self-defined modules

import module_load_data as load_data
import module_pac_functions as pacf
import module_detect_pac as detect_pac
import module_pac_plots as pac_plt

from bycycle.filt import lowpass_filter
from bycycle.features import compute_features


#%% CORRECT VERSION

### datastruct[subj][ch][ep][data]  

channels = 128
subjects = ['Chibi', 'George', 'Kin2', 'Su']

# condition time
eyes_open = [[40.91, 1245.01], [0.57, 663.66], [59.73, 1259.43], [91.49, 1289.02]]
eyes_closed = [[1439.47, 2635.60], [720.12, 1325.19], [1592.66,2799.70], [1681.38, 2889.52]] 

fs = 1000

# epoch length
epoch_len_seconds = 20 # in seconds
epoch_len = epoch_len_seconds * fs

num_epoch = 30 # 30 epochs available in shortest recording


datastruct = [None] * len(subjects)

for subj in range(len(subjects)): 
       
    # go to specific map
    os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab\neurotycho\anesthesia_sleep_task')    
    filename = os.path.join(os.getcwd(), subjects[subj])   
    os.chdir(filename)

    ch_counter = 0 
    datastruct_ch = [None] * channels

    for file_idx in glob.glob("ECoG_ch*.mat"):
#        ch = file_idx.split('_ch')[1].split('.mat')[0]
#        if ch.startswith('0'):
#            ch = ch.split('0')[1]
        
        # get data
        data = sio.loadmat(file_idx)
        data = next(v for k,v in data.items() if 'ECoGData' in k)
        
        # get only eyes_closed part of data
        data = np.squeeze(data)[
                int(eyes_closed[subj][0] * fs):int(eyes_closed[subj][1] * fs)
                ]
        
        # for every epoch of 20 seconds, write data to channel specific structure
        datastruct_ch[ch_counter] = [data[ep * epoch_len:ep * epoch_len + epoch_len] for ep in range(num_epoch)]
        
        # counter
        ch_counter = ch_counter + 1 
    
    # write data to full structure
    datastruct[subj] = datastruct_ch

del(datastruct_ch)
del(data)

#%% Save
os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab') 
np.save('datastruct', datastruct) 

#%% Load

os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab') 
datastruct = np.load('datastruct.npy', allow_pickle=True)  

#%% Check channels

subj = 3   
c = 0 
for ch in range(len(datastruct[subj])):
    
    plt.plot(np.nanmean(datastruct[subj][ch], axis=0))
    plt.show()
    
    c = c + 1
    



#%% Create dataframe that will be used throughout

# df = pd.DataFrame(np.nan, index=range(0,10), columns=['A'])

# find length of ch * tr
size = len(datastruct) * len(datastruct[0]) * len(datastruct[0][0])
    
# create dataframe with columns: subj & ch
features_df = pd.DataFrame(index=range(0,size), columns=['subj','ch', 'ep'])

num_epoch = 30

# ETC. BUT FOR NOW TRY TO FIRST LOAD ALL THREE SUBJ. 

# fill subj & ch
features_df['subj'] = [subj for subj in range(len(datastruct)) 
                        for ch in range(len(datastruct[subj]))
                        for ep in range(num_epoch)]
features_df['ch'] = [ch for subj in range(len(datastruct)) 
                        for ch in range(len(datastruct[subj]))
                        for ep in range(num_epoch)]
features_df['ep'] = [ep for subj in range(len(datastruct)) 
                        for ch in range(len(datastruct[subj]))
                        for ep in range(num_epoch)]

#%% Calculate largest peak in PSD using FOOF

# original is [4, 55], [2, 8], 4
freq_range = [4, 48] # for peak detection
bw_lims = [2, 6]
max_n_peaks = 5
freq_range_long = [4, 118] # to establish a more reliable slope
fs = 1000

pd.options.mode.chained_assignment = None

features_df = detect_pac.fooof_highest_peak_epoch_4_12_monkey(datastruct, fs, freq_range, bw_lims, max_n_peaks, freq_range_long, features_df)

#%% So now we have the peaks. Use those as input phase frequency for detecting PAC
    
amplitude_providing_band = [80, 150]; #80-125 Hz band as original

features_df = detect_pac.cal_pac_values_varphase_monkey(datastruct, amplitude_providing_band, fs, features_df)

features_df.to_csv('features_df.csv', sep=',', index=False)


#%% How many channels have PAC when channels are NOT resampled?

perc = features_df['pac_presence'][features_df['pac_presence'] == 1].sum()/ \
        len(features_df['pac_presence'])

print('in ', perc * 100, 'channels is PAC (unresampled)')


#%% Run resampling with variable phase and calculate true Z-values of PACs

# to ensure it does not give warnings when we change a specific value
# in a specific column 
pd.options.mode.chained_assignment = None

num_resamples = 300

features_df = detect_pac.resampled_pac_varphase_monkey(datastruct, amplitude_providing_band, fs, num_resamples, features_df)

features_df.to_csv('features_df.csv', sep=',', index=False)


#%% How many channels have PAC after resampling?

perc_resamp = features_df['resamp_pac_presence'][features_df['resamp_pac_presence'] == 1].sum()/ \
        len(features_df['resamp_pac_presence'])

print('in ', perc_resamp * 100, 'channels is PAC (resampled)')

# pac_idx = list(np.where(features_df['resamp_pac_presence'] == 1))
    
#%% Load all data that where created above to start ByCycle

# change dir 
os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab')

# data
datastruct = np.load('datastruct_full.npy', allow_pickle=True)
elec_locs = np.load('elec_locs.npy', allow_pickle=True)

# dataframe
features_df = pd.read_csv('features_df.csv', sep=',')

#%% Get bycycle features ATTENUATION PROBlEMS IN SOME CHANNELS

f_lowpass = 55
N_seconds = timewindow / num_epochs - 2

burst_kwargs = {'amplitude_fraction_threshold': 0.25,
                'amplitude_consistency_threshold': .4,
                'period_consistency_threshold': .45,
                'monotonicity_threshold': .6,
                'N_cycles_min': 3}

features_df['volt_amp'] = np.nan
features_df['rdsym']  = np.nan
features_df['ptsym']  = np.nan

# for every channel with pac
for ii in range(len(features_df)):
    
    # for every channel that has peaks
    if ~np.isnan(features_df['CF'][ii]):
    
        # define phase providing band
        CF = features_df['CF'][ii]
        BW = features_df['BW'][ii]
        
        phase_providing_band= [(CF - (BW/2)),  (CF + (BW/2))]
        
        subj = features_df['subj'][ii]
        ch = features_df['ch'][ii]
        ep = features_df['ep'][ii]
        data = datastruct[subj][ch][(ep*fs*epoch_len):((ep*fs*epoch_len)+fs*epoch_len)] 
        
        signal = lowpass_filter(data, fs, f_lowpass, N_seconds=N_seconds, remove_edge_artifacts=False)
        
        bycycle_df = compute_features(signal, fs, phase_providing_band,  burst_detection_kwargs=burst_kwargs)
        
        features_df['volt_amp'][ii] = bycycle_df['volt_peak'].median()
        features_df['rdsym'][ii] = bycycle_df['time_rdsym'].median()
        features_df['ptsym'][ii] = bycycle_df['time_ptsym'].median()
        
        print('this was ch', ii)
        
        
features_df.to_csv('features_df.csv', sep=',', index=False)


#%% 
# C:\Users\jaapv\Desktop\master\VoytekLab\neurotycho\anesthesia_sleep_task

