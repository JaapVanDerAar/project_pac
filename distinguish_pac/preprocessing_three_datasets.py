#%% imports

import os
import numpy as np
import scipy.io as sio
import pandas as pd


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
from module_load_data import get_signal


from bycycle.filt import lowpass_filter
from bycycle.features import compute_features

#%% Load human data

# metadata
subjects = ['al','ca','cc','de','fp','gc','gf','gw',
          'h0','hh','jc','jm','jp','mv','rh','rr',
          'ug','wc','wm','zt']
os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab')
dat_name = 'fixation_pwrlaw'
fs = 1000

# timewindow
timewindow = 100000/fs # 110800 is the length of shortest recording

# load data function for human   
datastruct_human, _ = load_data.load_data_human(dat_name, subjects, fs, timewindow)

#%% Load monkey data

# metadata
channels = 128
subjects = ['Chibi', 'George', 'Kin2', 'Su']
fs = 1000

# condition time
eyes_open = [[40.91, 1245.01], [0.57, 663.66], [59.73, 1259.43], [91.49, 1289.02]]
eyes_closed = [[1439.47, 2635.60], [720.12, 1325.19], [1592.66,2799.70], [1681.38, 2889.52]] 

# timewindow
epoch_len_seconds = 100 # in seconds
epoch_len = epoch_len_seconds * fs
num_epoch = 6 # shortest recording is 67000 (6 * 100s recording)

# load data function
datastruct_monkey = load_data.load_data_monkey(eyes_closed, subjects, fs, epoch_len, num_epoch, channels)

#%% Load rat data

# metadata
subjects = ['Bond', 'Dudley', 'Frank']
fs = 1500

# timewindow
epoch_len_seconds = 100 # in seconds
num_tetrodes_rat = 30
num_epoch_rat = 5

# load data function
datastruct_rat = load_data.load_data_rat(subjects, fs, epoch_len_seconds, num_epoch_rat, num_tetrodes_rat)

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

# monkey dataframe
num_epochs_monkey = 6
size_monkey = len(datastruct_monkey) * len(datastruct_monkey[0]) * len(datastruct_monkey[0][0])

features_df_monkey = pd.DataFrame(index=range(0,size_monkey), columns=['subj', 'day', 'ch', 'ep'])

# fill in dataframe with days always 0 
features_df_monkey['subj'] = [subj for subj in range(len(datastruct_monkey)) 
                        for ch in range(len(datastruct_monkey[subj]))
                        for ep in range(num_epochs_monkey)]
features_df_monkey['day'] = 0
features_df_monkey['ch'] = [ch for subj in range(len(datastruct_monkey)) 
                        for ch in range(len(datastruct_monkey[subj]))
                        for ep in range(num_epochs_monkey)]
features_df_monkey['ep'] = [ep for subj in range(len(datastruct_monkey)) 
                        for ch in range(len(datastruct_monkey[subj]))
                        for ep in range(num_epochs_monkey)]

# rat dataframe
num_epochs_rat = 5
num_tetrodes_rat = 30
size_rat = (len(datastruct_rat[0]) + len(datastruct_rat[1]) + len(datastruct_rat[2]))\
            * num_tetrodes_rat * num_epoch_rat
 
features_df_rat = pd.DataFrame(index=range(0,size_rat), columns=['subj','day', 'ch', 'ep'])

# fill in dataframe
features_df_rat['subj'] = [subj for subj in range(len(datastruct_rat)) 
                        for day in range(len(datastruct_rat[subj]))
                        for ch in range(num_tetrodes_rat)    
                        for ep in range(num_epochs_rat)]
features_df_rat['day'] = [day for subj in range(len(datastruct_rat)) 
                        for day in range(len(datastruct_rat[subj]))
                        for ch in range(num_tetrodes_rat)   
                        for ep in range(num_epochs_rat)]
features_df_rat['ch'] = [ch for subj in range(len(datastruct_rat)) 
                        for day in range(len(datastruct_rat[subj]))
                        for ch in range(num_tetrodes_rat)   
                        for ep in range(num_epochs_rat)]
features_df_rat['ep'] = [ep for subj in range(len(datastruct_rat)) 
                        for day in range(len(datastruct_rat[subj]))
                        for ch in range(num_tetrodes_rat)   
                        for ep in range(num_epochs_rat)]


#%% Put data and dataframe in dictionary and delete datastructs and dataframes

# create data dictionary
data_dict = {'human': datastruct_human, 'monkey': datastruct_monkey, 'rat': datastruct_rat}
features_df = {'human': features_df_human, 'monkey': features_df_monkey, 'rat': features_df_rat}

# delete datastructs
del(datastruct_human)
del(datastruct_monkey)
del(datastruct_rat)
del(features_df_human)
del(features_df_monkey)
del(features_df_rat)

# save data_dict (or load)
os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab')
np.save('data_dict', data_dict)
np.save('features_df', features_df) 

#%% Load data
os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab')
data_dict = np.load('data_dict.npy', allow_pickle=True).item()
features_df = np.load('features_df.npy', allow_pickle=True).item()


#%% Detect oscillatory peaks in PSD using FOOOF

freq_range = [4, 48] # for peak detection
bw_lims = [2, 6]
max_n_peaks = 5

pd.options.mode.chained_assignment = None

for key in data_dict: 
    
    features_df = detect_pac.find_fooof_peaks(data_dict, freq_range, bw_lims, max_n_peaks, features_df, key)


#%% Detect if there are oscillatory LF peaks
    
freq_range = [4, 100] # for peak detection
bw_lims = [2, 5]
max_n_peaks = 6

for key in data_dict:
    
    # works, but function is quite a mess
    features_df = detect_pac.find_fooof_peaks_lowgamma(data_dict, freq_range, bw_lims, max_n_peaks, features_df, key)

#%% Calculate PAC
    
amplitude_providing_band = [80, 250]; #80-125 Hz band as original

for key in data_dict:
    
    features_df = detect_pac.calculate_pac(data_dict, amplitude_providing_band, features_df, key)
            
        
#%% Calculate PAC for low gamma
    
for key in data_dict:   

    features_df = detect_pac.calculate_pac_lowgamma(data_dict, features_df, key)
    
#%% percentages of PAC low gamma

#len(features_df['rat'][features_df['rat']['LG_pac_presence']==1]) / (len(features_df['rat'][features_df['rat']['LG_pac_presence']==0]) + len(features_df['rat'][features_df['rat']['LG_pac_presence']==1])) 
#len(features_df['human'][features_df['human']['LG_pac_presence']==1]) / (len(features_df['human'][features_df['human']['LG_pac_presence']==0]) + len(features_df['human'][features_df['human']['LG_pac_presence']==1])) 
#len(features_df['monkey'][features_df['monkey']['LG_pac_presence']==1]) / (len(features_df['monkey'][features_df['monkey']['LG_pac_presence']==0]) + len(features_df['monkey'][features_df['monkey']['LG_pac_presence']==1])) 

#%% Bycycle 

# ByCycle parameters
f_lowpass = 45
N_seconds = 2
fs = 1000
burst_kwargs = {'amplitude_fraction_threshold': .3,
                'amplitude_consistency_threshold': .4,
                'period_consistency_threshold': .5,
                'monotonicity_threshold': .8,
                'N_cycles_min': 10}

# create empty dictionairy for burst list output 
burst_list_dict = {}

for key in data_dict:
    
    features_df[key]['volt_amp'] = np.nan
    features_df[key]['rdsym']  = np.nan
    features_df[key]['ptsym']  = np.nan
    
    burst_list = [np.nan] * len(features_df[key])
    
    # for every channel with pac
    for ii in range(len(features_df[key])):
        
        # for every channel that has peaks
        if ~np.isnan(features_df[key]['CF'][ii]):
        
            # define phase providing band
            CF = features_df[key]['CF'][ii]
            BW = features_df[key]['BW'][ii]
            
           # phase_providing_band= [(CF - (BW/2))-1,  (CF + (BW/2))+1]
            
            # only for now, delete with new FOOOF
            if BW < 5:
                phase_providing_band= [(CF - (BW/2))-1,  (CF + (BW/2))+1]
            else: 
                phase_providing_band= [(CF - (BW/2)),  (CF + (BW/2))]
            
            
            signal, fs = get_signal(data_dict, features_df, key, ii)    
            
            lp_signal = lowpass_filter(signal, fs, f_lowpass, N_seconds=N_seconds, remove_edge_artifacts=False)
            
            bycycle_df = compute_features(lp_signal, fs, phase_providing_band,  burst_detection_kwargs=burst_kwargs)
            
            features_df[key]['volt_amp'][ii] = bycycle_df['volt_peak'].median()
            features_df[key]['rdsym'][ii] = bycycle_df['time_rdsym'].median()
            features_df[key]['ptsym'][ii] = bycycle_df['time_ptsym'].median()
            
            # for finding longest streak
            
            # We have to ensure that the index is sorted
            bycycle_df.sort_index(inplace=True)
            # Resetting the index to create a column
            bycycle_df.reset_index(inplace=True)
            
            # create dataframe that only accumulates the true bursts
            streak_counter = bycycle_df.groupby((bycycle_df['is_burst']==False).cumsum()).agg({'index': ['count', 'min', 'max']})
            
            # Removing useless column level
            streak_counter.columns = streak_counter.columns.droplevel()
            
            # get maximum streak
            longest_streak = streak_counter[streak_counter['count']==streak_counter['count'].max()]
            
            # get starting and end point
            start_streak = longest_streak.iloc[0]['min'] + 1 
            end_streak = longest_streak.iloc[0]['max']
            
            biggest_burst_values = [bycycle_df['sample_last_trough'][start_streak],
                                    bycycle_df['sample_next_trough'][end_streak],
                                    list(bycycle_df['volt_peak'][start_streak:end_streak]),
                                    list(bycycle_df['time_rdsym'][start_streak:end_streak]),
                                    list(bycycle_df['time_ptsym'][start_streak:end_streak])]
            
            burst_list[ii] = biggest_burst_values 
            
            print('this was ch', ii)
            
    burst_list_dict[key] = burst_list
            
        
#%% Save features_df and burst_list_dict
# save data_dict (or load)
os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab')
np.save('features_df', features_df) 
np.save('burst_list_dict', burst_list_dict)