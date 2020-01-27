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


#%% Function + metadata including days (total of 4500)

subjects = ['Bond', 'Dudley', 'Frank']

# parameters
epoch_len_seconds = 100 # in seconds
num_tetrodes = 30
num_epoch = 5
fs = 1500


# length epoch
epoch_len = int(epoch_len_seconds * fs)

# create datastructure
datastruct = [None] * len(subjects)

# for every subject
for subj in range(len(subjects)): 

    # go to task map
    os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab\hippocampus_rats')    
    task_dir = os.path.join(os.getcwd(), subjects[subj])
    os.chdir(task_dir)
    
    # day specific datastructure
    datastruct_day = [None] * len(glob.glob("*task*.mat"))
    
    for day_counter in range(len(glob.glob("*task*.mat"))):
        
        # go to task map again
        os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab\hippocampus_rats')    
        task_dir = os.path.join(os.getcwd(), subjects[subj])
        os.chdir(task_dir)
    
        # get task file of the first recording day
        task_idx = glob.glob("*task*.mat")[day_counter]
        task = sio.loadmat(task_idx)
        
        # get day of the task
        day = task_idx.split('task')[1].split('.mat')[0]
        
        # if first task is sleep task
        if task['task'][0,int(day)-1][0,0][0,0][0][0] == 'sleep': 
            
            # go to the subject folder with the data
            os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab\hippocampus_rats')  
            filename = os.path.join(os.getcwd(), subjects[subj], 'EEG')   
            os.chdir(filename)
    
            
            # channel datastruct with length of number of channels    
            datastruct_ch = [None] * num_tetrodes
        
            # for each recording for that day
            for file_idx in glob.glob(('*eeg' + day)+'-1-*.mat'):
                
                # get data
                data = sio.loadmat(file_idx)
                
                # find channel by splitting file name -1 for index
                ch = int(file_idx.split('eeg' + day + '-1-')[1].split('.mat')[0]) - 1
                
                # get fs and start_time
                # fs = data['eeg'][0,int(day)-1][0,0][0,ch][0,0]['samprate'][0][0]
                # start_time = data['eeg'][0,int(day)-1][0,ep][0,ch_counter][0,0]['starttime'][0][0]     
                
                # squeeze data
                data = np.squeeze(data['eeg'][0,int(day)-1][0,0][0,ch][0,0]['data'])
    
                
                # for every epoch of 20 seconds, write data to channel specific structure
                datastruct_ch[ch] = [data[ep * epoch_len:ep * epoch_len + epoch_len] for ep in range(num_epoch)]

            
        
        # write channel data to day structure
        datastruct_day[day_counter] = datastruct_ch

            
    # write data to full structure
    datastruct[subj] = datastruct_day

del(datastruct_day)
del(datastruct_ch)
del(data)
del(task)

#%% Save
os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab') 
np.save('datastruct_rats', datastruct) 


#%% Load

os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab') 
datastruct = np.load('datastruct_rats.npy', allow_pickle=True)  
features_df = 


#%% Set up dataframe

# find length dataframe: total number of days across subjects * num_tetrodes * num_epochs
size = (len(datastruct[0]) + len(datastruct[1]) + len(datastruct[2]))\
            * num_tetrodes * num_epoch
            
    
# create dataframe with columns: subj, day, ch, & ep
features_df = pd.DataFrame(index=range(0,size), columns=['subj','day', 'ch', 'ep'])

# fill in dataframe
features_df['subj'] = [subj for subj in range(len(datastruct)) 
                        for day in range(len(datastruct[subj]))
                        for ch in range(num_tetrodes)    
                        for ep in range(num_epoch)]
features_df['day'] = [day for subj in range(len(datastruct)) 
                        for day in range(len(datastruct[subj]))
                        for ch in range(num_tetrodes)   
                        for ep in range(num_epoch)]
features_df['ch'] = [ch for subj in range(len(datastruct)) 
                        for day in range(len(datastruct[subj]))
                        for ch in range(num_tetrodes)   
                        for ep in range(num_epoch)]
features_df['ep'] = [ep for subj in range(len(datastruct)) 
                        for day in range(len(datastruct[subj]))
                        for ch in range(num_tetrodes)   
                        for ep in range(num_epoch)]


#%%

freq_range = [4, 48] # for peak detection
bw_lims = [2, 6]
max_n_peaks = 5
freq_range_long = [2, 118] # to establish a more reliable slope
fs = 1500

pd.options.mode.chained_assignment = None

features_df = detect_pac.fooof_highest_peak_epoch_rat(datastruct, fs, freq_range, bw_lims, max_n_peaks, freq_range_long, features_df)

features_df.to_csv('features_df_rat.csv', sep=',', index=False)

#%%

amplitude_providing_band = [65, 150]; #80-125 Hz band as original

features_df = detect_pac.cal_pac_values_varphase_rat(datastruct, amplitude_providing_band, fs, features_df)

features_df.to_csv('features_df_rat.csv', sep=',', index=False)


#%% How many channels have PAC when channels are NOT resampled?

perc = features_df['pac_presence'][features_df['pac_presence'] == 1].sum()/ \
        len(features_df['pac_presence'])

print('in ', perc * 100, 'channels is PAC (unresampled)')


#%% Run resampling with variable phase and calculate true Z-values of PACs

# to ensure it does not give warnings when we change a specific value
# in a specific column 
pd.options.mode.chained_assignment = None

num_resamples = 300

features_df = detect_pac.resampled_pac_varphase_rat(datastruct, amplitude_providing_band, fs, num_resamples, features_df)

features_df.to_csv('features_df_rats.csv', sep=',', index=False)

#%% Load

os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab') 
datastruct = np.load('datastruct_rats.npy', allow_pickle=True)  
features_df = pd.read_csv('features_df_rats.csv', sep=',')

#%% How many channels have PAC after resampling?

perc_resamp = features_df['resamp_pac_presence'][features_df['resamp_pac_presence'] == 1].sum()/ \
        len(features_df['resamp_pac_presence'])

print('in ', perc_resamp * 100, 'channels is PAC (resampled)')

#%% Get bycycle features ATTENUATION PROBlEMS IN SOME CHANNELS

# pac_idx = list(np.where(features_df['resamp_pac_presence'] == 1))


f_lowpass = 55
N_seconds = 2
fs = 1500

burst_kwargs = {'amplitude_fraction_threshold': .3,
                'amplitude_consistency_threshold': .4,
                'period_consistency_threshold': .5,
                'monotonicity_threshold': .8,
                'N_cycles_min': 5}

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
        
        # get data
        subj = features_df['subj'][ii]
        day = features_df['day'][ii]
        ch = features_df['ch'][ii]
        ep = features_df['ep'][ii]

        data = datastruct[subj][day][ch][ep]
        
        signal = lowpass_filter(data, fs, f_lowpass, N_seconds=N_seconds, remove_edge_artifacts=False)
        
        bycycle_df = compute_features(signal, fs, phase_providing_band,  burst_detection_kwargs=burst_kwargs)
        
        features_df['volt_amp'][ii] = bycycle_df['volt_peak'].median()
        features_df['rdsym'][ii] = bycycle_df['time_rdsym'].median()
        features_df['ptsym'][ii] = bycycle_df['time_ptsym'].median()
        
        print('this was ch', ii)
        
        
features_df.to_csv('features_df_rats.csv', sep=',', index=False)






