#%% Import standard libraries
import os
import numpy as np


#%% Import Voytek Lab libraries



#%% Change directory to import necessary modules

os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab\Code\distinguish_pac')

#%% import self-defined modules

import module_load_data as load_data
import module_pac_functions as pacf
import module_detect_pac as detect_pac
import module_pac_plots as pac_plt


from bycycle.filt import lowpass_filter
from bycycle.features import compute_features

import pandas as pd

#%% Meta data

subjects = ['al','ca','cc','de','fp','gc','gf','gw',
          'h0','hh','jc','jm','jp','mv','rh','rr',
          'ug','wc','wm','zt']

os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab')

dat_name = 'fixation_pwrlaw'

#%% Parameters 

fs = 1000

timewindow = 100000/fs # 110800 is the length of shortest recording

#%% Run load data middle timewindow function
   
datastruct, elec_locs = load_data.load_data_timewindow(dat_name, subjects, fs, timewindow)


#%% Save structure

np.save('datastruct_full', datastruct)
np.save('elec_locs', elec_locs)


#%% Create dataframe that will be used throughout

# df = pd.DataFrame(np.nan, index=range(0,10), columns=['A'])

# find length of subj * ch
size = 0
for subj in range(len(datastruct)): 
    size = size + len(datastruct[subj])
    
epoch_len = 20 # in seconds
num_epochs = int(timewindow/epoch_len)
    
# create dataframe with columns: subj & ch
features_df = pd.DataFrame(index=range(0,size*num_epochs), columns=['subj', 'ch', 'ep'])

# fill subj & ch
features_df['subj'] = [subj for subj in range(len(datastruct)) 
                        for ch in range(len(datastruct[subj]))
                        for ep in range(num_epochs)]
features_df['ch'] = [ch for subj in range(len(datastruct)) 
                        for ch in range(len(datastruct[subj]))
                        for ep in range(num_epochs)]
features_df['ep'] = [ep for subj in range(len(datastruct)) 
                        for ch in range(len(datastruct[subj]))
                        for ep in range(num_epochs)]


#%% Calculate largest peak in PSD using FOOF

# original is [4, 55], [2, 8], 4
freq_range = [4, 55] # for peak detection
bw_lims = [2, 6]
max_n_peaks = 5
freq_range_long = [4, 118] # to establish a more reliable slope

features_df = detect_pac.fooof_highest_peak_epoch_4_12(datastruct, epoch_len, fs, freq_range, bw_lims, max_n_peaks, freq_range_long, features_df)

#%% Write FOOOF features to dataframe

# ONLY NECESSARY FOR EARLIER VERSION OF FUNCTION ABOVE 

## periodic component
#features_df['CF'] = [
#        psd_peaks[features_df['subj'][ii]][features_df['ch'][ii]][features_df['ep'][ii]][0]
#        for ii in range(len(features_df))
#        ]
#features_df['Amp'] = [
#        psd_peaks[features_df['subj'][ii]][features_df['ch'][ii]][features_df['ep'][ii]][1] 
#        for ii in range(len(features_df))
#        ]
#features_df['BW'] = [
#        psd_peaks[features_df['subj'][ii]][features_df['ch'][ii]][features_df['ep'][ii]][2] 
#        for ii in range(len(features_df))
#        ]
#
## aperiodic component
#features_df['offset'] = [
#        backgr_params[features_df['subj'][ii]][features_df['ch'][ii]][features_df['ep'][ii]][0] 
#        for ii in range(len(features_df))
#        ]
#features_df['knee'] = [
#        backgr_params[features_df['subj'][ii]][features_df['ch'][ii]][features_df['ep'][ii]][1] 
#        for ii in range(len(features_df))
#        ]
#features_df['exp'] = [
#        backgr_params[features_df['subj'][ii]][features_df['ch'][ii]][features_df['ep'][ii]][2] 
#        for ii in range(len(features_df))
#        ]
#
## aperiodic component long frequency range
#features_df['offset_long'] = [
#        backgr_params_long[features_df['subj'][ii]][features_df['ch'][ii]][features_df['ep'][ii]][0] 
#        for ii in range(len(features_df))
#        ]
#features_df['knee_long'] = [
#        backgr_params_long[features_df['subj'][ii]][features_df['ch'][ii]][features_df['ep'][ii]][1] 
#        for ii in range(len(features_df))
#        ]
#features_df['exp_long'] = [
#        backgr_params_long[features_df['subj'][ii]][features_df['ch'][ii]][features_df['ep'][ii]][2] 
#        for ii in range(len(features_df))
#        ]


#%% So now we have the peaks. Use those as input phase frequency for detecting PAC
    
amplitude_providing_band = [62, 118]; #80-125 Hz band as original

features_df = detect_pac.cal_pac_values_varphase(datastruct, amplitude_providing_band, fs, features_df, epoch_len)

#%% How many channels have PAC when channels are NOT resampled?

perc = features_df['pac_presence'][features_df['pac_presence'] == 1].sum()/ \
        len(features_df['pac_presence'])

print('in ', perc * 100, 'channels is PAC (unresampled)')


#%% Run resampling with variable phase and calculate true Z-values of PACs

# to ensure it does not give warnings when we change a specific value
# in a specific column 
pd.options.mode.chained_assignment = None

num_resamples = 1000

features_df = detect_pac.resampled_pac_varphase(datastruct, amplitude_providing_band, fs, num_resamples, features_df, epoch_len)

features_df.to_csv('features_df_epoch_4_12.csv', sep=',', index=False)


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

