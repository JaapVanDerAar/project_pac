#%% imports 

import numpy as np
import os
import pandas as pd

os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab\Code\distinguish_pac')
from bycycle.filt import lowpass_filter
from bycycle.features import compute_features
from bycycle.burst import plot_burst_detect_params

#%% Load data 

# change dir 
os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab')

# data
datastruct = np.load('datastruct.npy', allow_pickle=True)

# dataframe
features_df = pd.read_csv('features_df.csv', sep=',')

#%% 




#%% parameters

fs = 1000
# Fs = 1000
# epoch length
epoch_len_seconds = 100 # in seconds

f_lowpass = 45
N_seconds = 2

#burst_kwargs = {'amplitude_fraction_threshold': 0.25,
#                'amplitude_consistency_threshold': .4,
#                'period_consistency_threshold': .45,
#                'monotonicity_threshold': .6,
#                'N_cycles_min': 3}


burst_kwargs = {'amplitude_fraction_threshold': .3,
                'amplitude_consistency_threshold': .4,
                'period_consistency_threshold': .5,
                'monotonicity_threshold': .8,
                'N_cycles_min': 10}

#%%

#        
#burst_dataframe = pd.DataFrame(index=range(0,len(features_df)), 
#                               columns=['start_sample','end_sample', 'volt_amp', 'rdsym','ptsym',])

burst_list = [None] * len(features_df)

 
# for every channel with pac
for ii in range(340,len(features_df)):
    
    # for every channel that has peaks
    if ~np.isnan(features_df['CF'][ii]):
    
        # define phase providing band
        CF = features_df['CF'][ii]
        BW = features_df['BW'][ii]
        
        # only for now, delete with new FOOOF
        if BW < 5:
            phase_providing_band= [(CF - (BW/2))-1,  (CF + (BW/2))+1]
        else: 
            phase_providing_band= [(CF - (BW/2)),  (CF + (BW/2))]
               
        subj = features_df['subj'][ii]
        ch = features_df['ch'][ii]
        ep = features_df['ep'][ii]
        data = datastruct[subj][ch][ep]
        
        signal = lowpass_filter(data, fs, f_lowpass, N_seconds=N_seconds, remove_edge_artifacts=False)
        
        bycycle_df = compute_features(signal, fs, phase_providing_band, burst_detection_kwargs=burst_kwargs)
#
#        plot_burst_detect_params(signal, fs, bycycle_df,
#                         burst_kwargs, tlims=(0, 5), figsize=(16, 3), plot_only_result=True)
#
#        plot_burst_detect_params(signal, fs, bycycle_df,
#                         burst_kwargs, tlims=(0, 5), figsize=(16, 3))
        
        # find biggest length with no violations   

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

# bandpass_filter(signal, fs, phase_providing_band, N_seconds=N_seconds, plot_frequency_response=True)
os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab') 
np.save('burst_list_monkey', burst_list) 



#%% 

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
        data = datastruct[subj][ch][ep]
        
        signal = lowpass_filter(data, fs, f_lowpass, N_seconds=N_seconds, remove_edge_artifacts=False)
        
        bycycle_df = compute_features(signal, fs, phase_providing_band, burst_detection_kwargs=burst_kwargs)

#        plot_burst_detect_params(signal, Fs, bycycle_df,
#                         burst_kwargs, tlims=(0, 5), figsize=(16, 3), plot_only_result=True)
#
#        plot_burst_detect_params(signal, Fs, bycycle_df,
#                         burst_kwargs, tlims=(0, 5), figsize=(16, 3))
        
        # find biggest length with no violations   

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
        
        bycycle_df[start_streak:end_streak]
   
        # for output dataframe
        burst_dataframe['start_sample'][ii] = bycycle_df['sample_last_trough'][start_streak]
        burst_dataframe['end_sample'][ii] = bycycle_df['sample_next_trough'][end_streak]
        burst_dataframe['volt_amp'][ii] = bycycle_df['volt_peak'][start_streak:end_streak].mean()
        burst_dataframe['rdsym'][ii] = bycycle_df['time_rdsym'][start_streak:end_streak].mean()
        burst_dataframe['ptsym'][ii] = bycycle_df['time_ptsym'][start_streak:end_streak].mean()

        
        plt.hist(plt.hist(bycycle_df['time_ptsym'][start_streak:end_streak]))















#%%

sample_last_trough
sample_next_trough
# min index - 1

plot_burst_detect_params(signal, Fs, bycycle_df,
                 burst_kwargs, tlims=(0, 5), figsize=(16, 3), plot_only_result=True)


#
from itertools import groupby
L = [1,1,1,-1,-1,1,-1,1,1,-1,-1,-1,1,-1]
max(sum(1 for i in g) for k,g in groupby(L))


[list(g) for k, g in groupby('AAAABBBCCD')] --> AAAA BBB CC D

test_list = [list(g) for k, g in groupby(test)]

xx = range(10)
yy = [0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
for group in groupby(iter(xx), lambda x: yy[x]):
    print(group[0], list(group[1]))
    

#%% 

df= pd.DataFrame([True, True, False, False, False, False, True, False, False], 
              index=pd.to_datetime(['2015-05-01', '2015-05-02', '2015-05-03',
                                   '2015-05-04', '2015-05-05', '2015-05-06',
                                   '2015-05-07', '2015-05-08', '2015-05-09']), 
              columns=['A'])

# We have to ensure that the index is sorted
df.sort_index(inplace=True)
# Resetting the index to create a column
df.reset_index(inplace=True)

# Grouping by the cumsum and counting the number of dates and getting their min and max
df = df.groupby(df['A'].cumsum()).agg(
    {'index': ['count', 'min', 'max']})

# Removing useless column level
df.columns = df.columns.droplevel()
#%%
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from bycycle.filt import lowpass_filter


signal = data
Fs = 1000
phase_providing_band = [(CF - (BW/2)),  (CF + (BW/2))]
f_lowpass = 30
N_seconds = .2

signal_low = lowpass_filter(signal, Fs, f_lowpass,
                            N_seconds=N_seconds[jj], remove_edge_artifacts=False,
                            plot_frequency_response=True)
plt.plot(signal_low[0:2000])







# Plot frequency response of bandpass filter
from bycycle.filt import bandpass_filter
test = lowpass_filter(signal, Fs, phase_providing_band, N_seconds=2, plot_frequency_response=True)
plt.plot(test[0:2000])

# Plot signal
t = np.arange(0, len(signal)/Fs, 1/Fs)
tlim = (2, 5)
tidx = np.logical_and(t>=tlim[0], t<tlim[1])

plt.figure(figsize=(12, 2))
plt.plot(t[tidx], signal[tidx], '.5')
plt.plot(t[tidx], signal_low[tidx], 'k')
plt.xlim(tlim)



#%% 
from bycycle.filt import bandpass_filter

# for attenuation problems

bandpass_filter(signal, Fs, (4, 10), N_seconds=.75, plot_frequency_response=True)
N_seconds = [0.1, 0.2, 0.5, 1, 2]
  

for jj in range(len(N_seconds)):
    
  # for every channel with pac
    for ii in range(10,15):
        
        # for every channel that has peaks
        if ~np.isnan(features_df['CF'][ii]):
        
            # define phase providing band
            CF = features_df['CF'][ii]
            BW = features_df['BW'][ii]
            
            print(CF)
            print(BW)
            
            phase_providing_band= [(CF - (BW/2)),  (CF + (BW/2))]
                   
            subj = features_df['subj'][ii]
            ch = features_df['ch'][ii]
            ep = features_df['ep'][ii]
            data = datastruct[subj][ch][ep]
            
            bandpass_filter(data, Fs, phase_providing_band, N_seconds=N_seconds[jj], plot_frequency_response=True)
    














