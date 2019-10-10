
#%% Packages

import os
import scipy.io
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert



#%% Load data

# set directory
os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab')

# subject label
subj = 'al'
subjLabel = subj + '_base'

# dataset
dataset = 'fixation_pwrlaw'
fs = 1000

# filename
filename = os.path.join(os.getcwd(), dataset, 'data', subjLabel)

# load data
dataStruct = sp.io.loadmat(filename)
data = dataStruct['data']
locs = dataStruct['locs']


#manually get only one channel
dataCh1 = data[:,0]



#%% Filtering functions

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



#%% Selecting the frequency bands 
phase_providing_band = [4,8]; #4-8 Hz band
amplitude_providing_band = [80, 125]; #80-125 Hz band


#calculating phase of theta
phase_data = butter_bandpass_filter(dataCh1, phase_providing_band[0], phase_providing_band[1], round(float(fs)));
phase_data_hilbert = hilbert(phase_data);
phase_data_angle = np.angle(phase_data_hilbert);

#calculating amplitude envelope of high gamma
amp_data = butter_bandpass_filter(dataCh1, amplitude_providing_band[0], amplitude_providing_band[1], round(float(fs)));
amp_data_hilbert = hilbert(amp_data);
amp_data_abs = abs(amp_data_hilbert);


#%% 2 seconds of the signal, method 1 

# plot both the theta signal and the phase [enlarged for visualisation]
# and the high gamma absolute signal/amplitude envelope
fig = plt.figure(figsize=(16,12))
plt.plot((phase_data_hilbert[0:2000]*0.1),label= 'Theta Signal')
plt.plot(phase_data_angle[0:2000],label= 'Phase of Theta')
plt.plot(amp_data_abs[0:2000],label= 'Amplitude of High Gamma')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))


#%% 2 seconds of the signal, method 2

#let's look at a small chunk of our data
plt.figure(figsize = (15,6));
plt.plot((dataCh1[1:int(fs)*2]-np.mean(data[1:int(fs)*2]))/np.std(dataCh1[1:int(fs)*2]),label= 'Raw Data'); #normalized raw data
plt.plot(phase_data_angle[1:int(fs)*2],label= 'Phase of Theta');
plt.plot(amp_data_abs[1:int(fs)*2]*0.02,label= 'Amplitude of High Gamma'); 
plt.xlabel('Two Seconds of Theta Phase and High Gamma Amplitude')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))


#%% Function for circle correlation
def circCorr(ang,line):
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

#calculate circle correlation
print(circCorr(phase_data_angle, amp_data_abs))

#%% Calculate the amplitude per phase bin 

#this takes a while to run. 
#If you make bin_size larger (36) it will run much quicker but the effect won't be as clear.
bin_size = 5; 
bins = range(-180,180+bin_size,bin_size); 
bins = np.dot(bins, 0.0174532925);

amps = [];

#filling phase bins with amplitudes
for x in range(len(bins)-1):
    # find the lower bound of the bin
    amps_above_lo_bound = np.where(phase_data_angle >= bins[x])[0];
    # find the higher bound of the bin
    amps_below_hi_bound = np.where(phase_data_angle < bins[x+1])[0];
    amps_below_hi_bound = set(amps_below_hi_bound);
    # select all samples that are within the range
    amp_inds_in_this_bin = [amp_val for amp_val in amps_above_lo_bound if amp_val in amps_below_hi_bound]
    # find corresponding amplitudes for these samples
    amps_in_this_bin = amp_data_abs[amp_inds_in_this_bin];
    # calculate mean
    amps.append(np.mean(amps_in_this_bin));

bins = bins[:len(bins)-1];

#normalizing to make the effect more clear
amps = (amps-np.mean(amps))/np.std(amps);

#%% Plot PAC figure

#plotting figure;
fig = plt.figure(figsize=(8,8))
ax = fig.add_axes([0.1, 0.1, 1, 1], polar=True)
ax.bar(bins, amps, width=bins[1]-bins[0], bottom=0.0)
plt.title('Phase Amplitude Coupling');


  
