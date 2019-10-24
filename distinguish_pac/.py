#%% lists 

# <list_name> = [<value>, <value>]

# can hold different types of data
#%% imports  


import os

import scipy.io as sio

#%% subject list and directory


#%% loop to load data in structure


# other inputs 

subjects = ['al','ca','cc','de','fp','gc','gf','gw',
          'h0','hh','jc','jm','jp','mv','rh','rr',
          'ug','wc','wm','zt']

dat_dir = '/Users/SP/Downloads'

dat_name = 'fixation_pwrlaw'

fs = 1000

timewindow = 5 # how many data want to select (in sec)

ss = 60 # where to start to select data per recording (in sec)



#%% Load datastructure function

def load_data_and_locs(dat_dir, dat_name, subjects, fs, timewindow, ss):
    """
    This function load the ECoG data from Kai Miller's database 
    Give following inputs:
    -   directory and name of data
    -   list of subjects 
    -   sampling frequency
    -   timewindow of which data to include
    -   starting point from where to extract data (in seconds)    
    """" 
    
    ss = ss * fs
    
    tw = timewindow * fs
    
    os.chdir(dat_dir)

    datastruct = [] 
    elec_locs = []
        
    for subj in range(len(subjects)):
        
        # get the filename
        sub_label = subjects[subj] + '_base'
        filename = os.path.join(os.getcwd(), dat_name, 'data', sub_label)
    
        # load data
        dataStruct = sio.loadmat(filename)
        data = dataStruct['data']
        locs = dataStruct['locs']
    
    
        # create empty list for subj data
        subj_data = []
        
        
        # for every channel in this subj
        for ch in range(len(data[0])):
            
            # extract data of specified timewindow
            ch_data = data[ss:ss+tw,ch]
            
            # store channel data in sub_data
            subj_data.append(ch_data)
            
        
        datastruct.append(subj_data)
        
        elec_locs.append(locs)
        

    return datastruct, elec_locs


#%% 
    
datastruct, elec_locs = load_data_and_locs(dat_dir, dat_name, subjects, fs, timewindow, ss)




#%% Imports 

import scipy.io
import scipy as sp
import numpy as np
from scipy.signal import butter, filtfilt, hilbert
import pandas as pd


#%% Filtering and circle correlation functions

def butter_bandpass(lowcut, highcut, fs, order=4):
    #lowcut is the lower bound of the frequency that we want to isolate
    #hicut is the upper bound of the frequency that we want to isolate
    #fs is the sampling rate of our data
    nyq = 0.5 * fs #nyquist frequency - see http://www.dspguide.com/ if you want more info
    low = float(lowcut) / nyq
    high = float(highcut) / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(mydata, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, mydata)
    return y

def circCorr(ang,line):
    # give amplitude and phase data as input, returns correlation stats
    n = len(ang)
    rxs = sp.stats.pearsonr(line,np.sin(ang))
    rxs = rxs[0]
    rxc = sp.stats.pearsonr(line,np.cos(ang))
    rxc = rxc[0]
    rcs = sp.stats.pearsonr(np.sin(ang),np.cos(ang))
    rcs = rcs[0]
    rho = np.sqrt((rxc**2 + rxs**2 - 2*rxc*rxs*rcs)/(1-rcs**2)) #r
    r_2 = rho**2 #r squared
    pval = 1- sp.stats.chi2.cdf(n*(rho**2),1)
    standard_error = np.sqrt((1-r_2)/(n-2))

    return rho, pval, r_2,standard_error



#%% Detect PAC parameters 
    
phase_providing_band = [4,8]; #4-8 Hz band
amplitude_providing_band = [80, 125]; #80-125 Hz band

#%% 


def detect_pac(datastruct, phase_providing_band, amplitude_providing_band, fs):
        
    # create output matrix of 20 * 64 (subj * max channels)
    pac_presence = pd.DataFrame(np.nan, index=range(len(datastruct)), columns=range(64))
    pac_pvals = pd.DataFrame(np.nan, index=range(len(datastruct)), columns=range(64))
    pac_rhos = pd.DataFrame(np.nan, index=range(len(datastruct)), columns=range(64))

    # for every subject
    for subj in range(len(datastruct)):
        
        # for every channel
        for ch in range(len(datastruct[subj])):
            
            data = datastruct[subj][ch] 
            
            #calculating phase of theta
            phase_data = butter_bandpass_filter(data, phase_providing_band[0], phase_providing_band[1], round(float(fs)));
            phase_data_hilbert = hilbert(phase_data);
            phase_data_angle = np.angle(phase_data_hilbert);
            
            #calculating amplitude envelope of high gamma
            amp_data = butter_bandpass_filter(data, amplitude_providing_band[0], amplitude_providing_band[1], round(float(fs)));
            amp_data_hilbert = hilbert(amp_data);
            amp_data_abs = abs(amp_data_hilbert);
            
            
            PAC_values = circCorr(phase_data_angle[2000:3000], amp_data_abs[2000:3000])
        
            if PAC_values[1] <= 0.05:
                
                pac_presence[ch][subj] = 1
                
            elif PAC_values[1] > 0.05: 
    
                pac_presence[ch][subj] = 0
                
            pac_pvals[ch][subj] = PAC_values[1]
            pac_rhos[ch][subj] = PAC_values[0]
                
        print('another one is done =), this was subj', subj)
    
    return pac_presence, pac_pvals, pac_rhos


#%% 
    
pac_presence, pac_pvals, pac_rhos = detect_pac(datastruct, phase_providing_band, amplitude_providing_band, fs)

np.save('pac_presence', pac_presence)

np.save('pac_pvals', pac_pvals)

np.save('pac_rhos', pac_rhos)
#%% Statistical resampling


num_resampl = 1000
# bands 

resampled_pvalues_subjlevel = []
resampled_rhovalues_subjlevel = []


# for every subject
for subj in range(len(datastruct)):

    # create datastructs on channel level to save resamples PAC values
    resampled_pvalues_channel = []
    resampled_rhovalues_channel = []
    
    # for every channel
    for ch in range(len(datastruct[subj])):
    
        # get data of that channel
        data = datastruct[subj][ch] 
        
        # create array of random numbers between 0 and total samples 
        # which is as long as the number of resamples
        roll_array = np.random.randint(0, len(data), size=num_resampl)
        
        resampled_pvalues_sample = []
        resampled_rhovalues_sample = []
     
        # for every resample
        for ii in range(len(roll_array)):

            # roll the phase data for a random amount 
            phase_data_roll = np.roll(data, roll_array[ii])
            
            #calculating phase of theta
            phase_data = butter_bandpass_filter(phase_data_roll, phase_providing_band[0], phase_providing_band[1], round(float(fs)));
            phase_data_hilbert = hilbert(phase_data);
            phase_data_angle = np.angle(phase_data_hilbert);
            
            #calculating amplitude envelope of high gamma
            amp_data = butter_bandpass_filter(data, amplitude_providing_band[0], amplitude_providing_band[1], round(float(fs)));
            amp_data_hilbert = hilbert(amp_data);
            amp_data_abs = abs(amp_data_hilbert);
    
            # calculate PAC
            PAC_values = circCorr(phase_data_angle[2000:3000], amp_data_abs[2000:3000])
            
            resampled_pvalues_sample.append(PAC_values[1])
            resampled_rhovalues_sample.append(PAC_values[0])
            
        resampled_pvalues_channel.append(resampled_pvalues_sample)
        resampled_rhovalues_channel.append(resampled_rhovalues_sample)
   
        print('this was ch', ch)
    
    
    resampled_pvalues_subjlevel.append(resampled_pvalues_channel)
    resampled_rhovalues_subjlevel.append(resampled_rhovalues_channel)     
    
    print('another one is done =), this was subj', subj)
        
        
        
 
#%% Save stuff

       
np.save('resampled_rhovalues_subjlevel', resampled_rhovalues_subjlevel)

np.save('resampled_pvalues_subjlevel', resampled_pvalues_subjlevel)


#%% imports

import matplotlib.pyplot as plt
#%% plots 

subj = 1 

for ch in range(len(resampled_rhovalues_subjlevel[subj])): 
    
    plt.figure(ch)
    a = plt.hist(resampled_rhovalues_subjlevel[1][ch], bins = 20)
    
            
            
#%% 
          
subj = 3
    
    
    
for ch in range(len(resampled_rhovalues_subjlevel[0])):
    
    print("""
    """ )
        
    true_z = (pac_rhos[ch][subj] - np.mean(resampled_rhovalues_subjlevel[subj][ch])) / np.std(resampled_rhovalues_subjlevel[subj][ch])
    print(true_z)
    
    
    if true_z > 0: 
        p_value = scipy.stats.norm.sf((true_z))
        print(p_value)
    
    
#%% Write true p-values into matrix
        

pac_true_pvals = pd.DataFrame(np.nan, index=range(len(resampled_pvalues_subjlevel)), columns=range(64))
pac_true_zvals = pd.DataFrame(np.nan, index=range(len(resampled_pvalues_subjlevel)), columns=range(64))
pac_true_presence = pd.DataFrame(np.nan, index=range(len(resampled_pvalues_subjlevel)), columns=range(64))


# for every subj
for subj in range(len(resampled_rhovalues_subjlevel)):
    
    # for every channel
    for ch in range(len(resampled_rhovalues_subjlevel[subj])):
        
        true_z = (pac_rhos[ch][subj] - np.mean(resampled_rhovalues_subjlevel[subj][ch])) / np.std(resampled_rhovalues_subjlevel[subj][ch])
        p_value = scipy.stats.norm.sf(abs(true_z))
            
        pac_true_pvals[ch][subj] = p_value
        pac_true_zvals[ch][subj] = true_z
        
        if pac_true_zvals[ch][subj] >= 0  and  pac_true_pvals[ch][subj] <= 0.05:

            pac_true_presence[ch][subj] = 1         
            
        else: 
        
            pac_true_presence[ch][subj] = 0 
        

#%% Save
            
np.save('pac_true_pvals', pac_true_pvals)

np.save('pac_true_zvals', pac_true_zvals)

np.save('pac_true_presence', pac_true_presence)
        

#%% 

a = (pac_true_presence == 1).sum()
a.sum()

#%% Plot
 


subj = 0 
ch = 0

plt_time = [2, 3] # time over which pac is also calculated
fs = 1000

#calculating phase of theta
phase_data = pacf.butter_bandpass_filter(datastruct[subj][ch], phase_providing_band[0], phase_providing_band[1], round(float(fs)));
phase_data_hilbert = hilbert(phase_data);
phase_data_angle = np.angle(phase_data_hilbert);

#calculating amplitude envelope of high gamma
amp_data = pacf.butter_bandpass_filter(datastruct[subj][ch], amplitude_providing_band[0], amplitude_providing_band[1], round(float(fs)));
amp_data_hilbert = hilbert(amp_data);
amp_data_abs = abs(amp_data_hilbert);

plt.figure(figsize = (20,8));
plt.plot((datastruct[subj][ch][plt_time[0]*fs:plt_time[1]*fs]),label= 'Raw Signal')
plt.plot((amp_data_hilbert[plt_time[0]*fs:plt_time[1]*fs]),label= 'High Gamma [80-125 Hz]')
plt.plot((phase_data_hilbert[plt_time[0]*fs:plt_time[1]*fs]),label= 'Theta [4-8 Hz]')

plt.xlabel('Two Seconds of Theta Phase, High Gamma Amplitude, and raw signal')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
           

#%%


pac_idx = list(np.where(pac_true_presence == 1))

# iterate with range(len(index))---> index[0][pieai] index[1][pieai]

#%% plot channels with PAC

plt_time = [2, 3] # time over which pac is also calculated
fs = 1000

for ix in range(len(pac_idx)):
        
#%% 
    
    ix += 1 
    
    subj = pac_idx[0][ix] 
    ch = pac_idx[1][ix]
    
    #calculating phase of theta
    phase_data = butter_bandpass_filter(datastruct[subj][ch], phase_providing_band[0], phase_providing_band[1], round(float(fs)));
    phase_data_hilbert = hilbert(phase_data);
    phase_data_angle = np.angle(phase_data_hilbert);
    
    #calculating amplitude envelope of high gamma
    amp_data = butter_bandpass_filter(datastruct[subj][ch], amplitude_providing_band[0], amplitude_providing_band[1], round(float(fs)));
    amp_data_hilbert = hilbert(amp_data);
    amp_data_abs = abs(amp_data_hilbert);
    
    
    
    plt.figure(figsize = (20,8));
    plt.plot((datastruct[subj][ch][plt_time[0]*fs:plt_time[1]*fs]),label= 'Raw Signal')
    plt.plot((amp_data_hilbert[plt_time[0]*fs:plt_time[1]*fs]),label= 'High Gamma [80-125 Hz]')
    plt.plot((phase_data_hilbert[plt_time[0]*fs:plt_time[1]*fs]),label= 'Theta [4-8 Hz]')
    
    plt.xlabel('Two Seconds of Theta Phase, High Gamma Amplitude, and raw signal')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
    
