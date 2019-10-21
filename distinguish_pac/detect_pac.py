#%% import packages

import os
import scipy.io
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert


from neurodsp.utils import create_times
from neurodsp.plts import plot_time_series, plot_power_spectra, plot_spectral_hist, plot_scv
from neurodsp.plts import plot_scv_rs_lines, plot_scv_rs_matrix 
from neurodsp.plts.time_series import plot_instantaneous_measure
from neurodsp.timefrequency import amp_by_time, phase_by_time
from neurodsp import spectral




#%% METADATA

# Load data and plot

subjects=['al','ca','cc','de','fp','gc','gf','gw',
          'h0','hh','jc','jm','jp','mv','rh','rr',
          'ug','wc','wm','zt']

os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab')

# dataset
dataset = 'fixation_pwrlaw'
fs = 1000

# subj info
subj = 16
ch = 17

# get the filename
sub_label = subjects[subj] + '_base'
filename = os.path.join(os.getcwd(), dataset, 'data', sub_label)

# load data
dataStruct = sp.io.loadmat(filename)
data = dataStruct['data']
locs = dataStruct['locs']

sig = data[:,ch]

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

#%% Selecting frequency bands 
    
phase_providing_band = [4,8]; #4-8 Hz band
amplitude_providing_band = [80, 125]; #80-125 Hz band


#%% Loop through every subj and channel to find which have PAC

# create output matrix of 20 * 64 (subj * channels)
PAC_presence = np.full((20,64),np.nan)

# for every subject
for subj in range(len(subjects)): 
    
    # get the filename
    sub_label = subjects[subj] + '_base'
    filename = os.path.join(os.getcwd(), dataset, 'data', sub_label)
    
    # load data
    dataStruct = sp.io.loadmat(filename)
    data = dataStruct['data']
    locs = dataStruct['locs']
    
    # for every channel 
    for ch in range(len(locs)):
        
        
    
        #calculating phase of theta
        phase_data = butter_bandpass_filter(data[50000:70000,ch], phase_providing_band[0], phase_providing_band[1], round(float(fs)));
        phase_data_hilbert = hilbert(phase_data);
        phase_data_angle = np.angle(phase_data_hilbert);
        
        #calculating amplitude envelope of high gamma
        amp_data = butter_bandpass_filter(data[50000:70000,ch], amplitude_providing_band[0], amplitude_providing_band[1], round(float(fs)));
        amp_data_hilbert = hilbert(amp_data);
        amp_data_abs = abs(amp_data_hilbert);


        PAC_values = circCorr(phase_data_angle[10000:12000], amp_data_abs[10000:12000])
        
        if PAC_values[1] <= 0.05:
            
            PAC_presence[subj, ch] = 1
            
        elif PAC_values[1] > 0.05: 

            PAC_presence[subj, ch] = 0
            
            
    print('another one is done =), this was subj', subj)
 

#%% Save and Load            
#            
#np.save('PAC_presence_2s6062.npy', PAC_presence)   
#
#PAC_pres = np.load('PAC_presence.npy')         
#     
        
#%% Amount of channels in which PAC is detected    

(PAC_presence == 1).sum() / ((PAC_presence == 1).sum() + (PAC_presence == 0).sum()) * 100



    
    
#%% Plot some data

# sub data
subj = 16
ch = 17

# time plot in seconds
plt_time = [60, 62]


# get the filename
sub_label = subjects[subj] + '_base'
filename = os.path.join(os.getcwd(), dataset, 'data', sub_label)

# load data
dataStruct = sp.io.loadmat(filename)
data = dataStruct['data']
locs = dataStruct['locs']

#calculating phase of theta
phase_data = butter_bandpass_filter(data[:,ch], phase_providing_band[0], phase_providing_band[1], round(float(fs)));
phase_data_hilbert = hilbert(phase_data);
phase_data_angle = np.angle(phase_data_hilbert);

#calculating amplitude envelope of high gamma
amp_data = butter_bandpass_filter(data[:,ch], amplitude_providing_band[0], amplitude_providing_band[1], round(float(fs)));
amp_data_hilbert = hilbert(amp_data);
amp_data_abs = abs(amp_data_hilbert);

PAC_values = circCorr(phase_data_angle, amp_data_abs)


# one way to visualize it using plt.plot
plt.figure(figsize = (20,8));
plt.plot((data[plt_time[0]*fs:plt_time[1]*fs,ch]),label= 'Raw Signal')
plt.plot((amp_data_hilbert[plt_time[0]*fs:plt_time[1]*fs]),label= 'High Gamma [80-125 Hz]')
plt.plot((phase_data_hilbert[plt_time[0]*fs:plt_time[1]*fs]),label= 'Theta [4-8 Hz]')

plt.xlabel('Two Seconds of Theta Phase, High Gamma Amplitude, and raw signal')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# other way using neurodsp.plts
times = create_times(len(sig)/fs, fs)
times = times[0:len(times)-1]
plot_time_series(times, [sig, amp_data_hilbert, phase_data_hilbert], ['Raw', 'Ampls', 'Phase'], xlim=plt_time)



#%% SPECTRAL POWER 

freq_mean, psd_mean = spectral.compute_spectrum(sig, fs, method='welch', avg_type='mean', nperseg=fs*2)

# Median of spectrogram ("median Welch")
freq_med, psd_med = spectral.compute_spectrum(sig, fs, method='welch', avg_type='median', nperseg=fs*2)

# Median filtered spectrum
freq_mf, psd_mf = spectral.compute_spectrum(sig, fs, method='medfilt')


# Plot the power spectra
plot_power_spectra([freq_mean[:200], freq_med[:200], freq_mf[100:10000]],
                   [psd_mean[:200], psd_med[:200], psd_mf[100:10000]],
                   ['Welch', 'Median Welch', 'Median Filter FFT'])


#%% SPECTRAL HISTOGRAM

freqs, bins, spect_hist = spectral.compute_spectral_hist(sig, fs, nbins=50, f_range=(0, 80),
                                                         cut_pct=(0.1, 99.9))

# Calculate a power spectrum, with median Welch
freq_med, psd_med = spectral.compute_spectrum(sig, fs, method='welch',
                                              avg_type='median', nperseg=fs*2)

# Plot the spectral histogram
plot_spectral_hist(freqs, bins, spect_hist, freq_med, psd_med)


#%% SPECTRAL COEFFICIENT OF VARIATION

freqs, scv = spectral.compute_scv(sig, fs, nperseg=int(fs), noverlap=0)

# Plot the SCV
plot_scv(freqs, scv)


# Bootstrapped

# Calculate SCV with the resampling method
freqs, t_inds, scv_rs = spectral.compute_scv_rs(sig, fs, nperseg=fs, method='bootstrap',
                                                rs_params=(20, 200))
# Plot the SCV, from the resampling method
plot_scv_rs_lines(freqs, scv_rs)

# Matrix

# Calculate SCV with the resampling method
freqs, t_inds, scv_rs = spectral.compute_scv_rs(sig, fs, method='rolling', rs_params=(10, 2))

# Plot the SCV, from the resampling method
plot_scv_rs_matrix(freqs, t_inds, scv_rs)

#%% TIME-FREQUENCY ANALYSIS

# Set the frequency range to be used
f_range = (13, 30)


#%% INSTANTANEOUS PHASE

# Compute instaneous phase from a signal
pha = phase_by_time(sig, fs, f_range)


# Plot example signal
_, axs = plt.subplots(2, 1, figsize=(15, 6))
plot_time_series(times, sig, xlim=plt_time, xlabel=None, ax=axs[0])
plot_instantaneous_measure(times, pha, xlim=plt_time, ax=axs[1])


#%% INSTATANEOUS AMPLITUDE

# Compute instaneous amplitude from a signal
amp = amp_by_time(sig, fs, f_range)

# Plot example signal
_, axs = plt.subplots(2, 1, figsize=(15, 6))
plot_instantaneous_measure(times, [sig, amp], 'amplitude',
                           labels=['Raw Voltage', 'Amplitude'],
                           xlim=[4, 5], xlabel=None, ax=axs[0])
plot_instantaneous_measure(times, [sig_filt_true, amp], 'amplitude',
                           labels=['Raw Voltage', 'Amplitude'], colors=['b', 'r'],
                           xlim=[4, 5], ax=axs[1])

#%% INSTANTANEUOUS FREQUENCY