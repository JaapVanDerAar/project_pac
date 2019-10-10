#%%

#importing modules
import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy.io
import scipy.signal
from scipy.signal import butter, filtfilt, hilbert

#loading ECoG data from local save
filename = 'emodat.mat'
filename = os.path.join('./', filename)
data = sp.io.loadmat(filename)
srate = data['srate'];
data = data['data']; # time series
data = data[0, :];

#%%
#filtering functions
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

#%%

#parameters
phase_providing_band = [4,8]; #4-8 Hz band
amplitude_providing_band = [80, 125]; #80-125 Hz band

#calculating phase of theta
phase_data = butter_bandpass_filter(data, phase_providing_band[0], phase_providing_band[1], round(float(srate)));
phase_data1 = hilbert(phase_data);
phase_data = angle(phase_data1);

#calculating amplitude envelope of high gamma
amp_data = butter_bandpass_filter(data, amplitude_providing_band[0], amplitude_providing_band[1], round(float(srate)));
amp_data = hilbert(amp_data);
amp_data = abs(amp_data);

#let's look at a small chunk of our data
figure(figsize = (15,6));
plt.plot((data[1:250]-mean(data[1:int(srate)*2]))/std(data[1:int(srate)*2]),label= 'Raw Data'); #normalized raw data
plt.plot((phase_data1[1:250]*0.1),label= 'Phase of Theta');
plt.plot(phase_data[1:250],label= 'Theta');
plt.plot(amp_data[1:250],label= 'Amplitude of High Gamma'); 
xlabel('Two Seconds of Theta Phase and High Gamma Amplitude')
legend(loc='center left', bbox_to_anchor=(1, 0.5))

#%%
def circCorr(ang,line):
    n = len(ang)
    rxs = sp.stats.pearsonr(line,sin(ang))
    rxs = rxs[0]
    rxc = sp.stats.pearsonr(line,cos(ang))
    rxc = rxc[0]
    rcs = sp.stats.pearsonr(sin(ang),cos(ang))
    rcs = rcs[0]
    rho = sqrt((rxc**2 + rxs**2 - 2*rxc*rxs*rcs)/(1-rcs**2)) #r
    r_2 = rho**2 #r squared
    pval = 1- sp.stats.chi2.cdf(n*(rho**2),1)
    standard_error = sqrt((1-r_2)/(n-2))

    return rho, pval, r_2,standard_error

print(circCorr(phase_data, amp_data))

#%%

#this takes a while to run. 
#If you make bin_size larger (36) it will run much quicker but the effect won't be as clear.
bin_size = 5; 
bins = range(-180,180+bin_size,bin_size); 
bins = dot(bins, 0.0174532925);

amps = [];

#filling phase bins with amplitudes
for x in range(len(bins)-1):
    amps_above_lo_bound = where(phase_data >= bins[x])[0];
    amps_below_hi_bound = where(phase_data < bins[x+1])[0];
    amps_below_hi_bound = set(amps_below_hi_bound);
    amp_inds_in_this_bin = [amp_val for amp_val in amps_above_lo_bound if amp_val in amps_below_hi_bound]
    amps_in_this_bin = amp_data[amp_inds_in_this_bin];
    amps.append(mean(amps_in_this_bin));

bins = bins[:len(bins)-1];

#normalizing to make the effect more clear
amps = (amps-mean(amps))/std(amps);

#plotting figure;
fig = figure(figsize=(8,8))
ax = fig.add_axes([0.1, 0.1, 1, 1], polar=True)
ax.bar(bins, amps, width=bins[1]-bins[0], bottom=0.0)
title('Phase Amplitude Coupling');

