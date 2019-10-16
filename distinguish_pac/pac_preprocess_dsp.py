
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
from neurodsp.timefrequency import amp_by_time, freq_by_time, phase_by_time
from neurodsp import spectral
from neurodsp.filt.filter import filter_signal



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

#%% Setting parameters

# Selecting frequency bands 
phase_providing_band = [4,8]; #4-8 Hz band
amplitude_providing_band = [80, 125]; #80-125 Hz band

# time plot in seconds
plt_time = [60, 62]

# create times structure for plotting
times = create_times(len(sig)/fs, fs)
times = times[0:len(times)-1]

#%% Calculate and plot 

# filter data on the providing bands
phase_filt_signal = filter_signal(data[:,ch], fs,'bandpass', phase_providing_band)
ampl_filt_signal = filter_signal(data[:,ch], fs, 'bandpass', amplitude_providing_band)

# calculate the phase
phase_signal = phase_by_time(data[:,ch], fs, phase_providing_band)


# Compute instaneous amplitude from a signal
amp_signal = amp_by_time(sig, fs, amplitude_providing_band)


#%% Plot the phase

# plot of phase: raw data, filtered, and phase
_, axs = plt.subplots(3, 1, figsize=(15, 6))
plot_time_series(times, data[:,ch], xlim=plt_time, xlabel=None, ax=axs[0])
plot_time_series(times, phase_filt_signal, xlim=plt_time, xlabel=None, ax=axs[1])
plot_instantaneous_measure(times, phase_signal, xlim=plt_time, ax=axs[2])



#%% Plot the amplitudes

# plot of amplitude: raw + abs ampls, and filtered + abs ampls
_, axs = plt.subplots(2, 1, figsize=(15, 6))
plot_instantaneous_measure(times, [data[:,ch], amp_signal], 'amplitude',
                           labels=['Raw Voltage', 'Amplitude'],
                           xlim=plt_time, xlabel=None, ax=axs[0])
plot_instantaneous_measure(times, [ampl_filt_signal, amp_signal], 'amplitude',
                           labels=['Raw Voltage', 'Amplitude'], colors=['b', 'r'],
                           xlim=plt_time, ax=axs[1])



#%% Calculate PAC

PAC_values = circCorr(phase_signal[60000:62000], amp_signal[60000:62000])


#%% Two ways of visualizing signal (raw, theta, and high gamma)

plt.figure(figsize = (20,8));
plt.plot((data[plt_time[0]*fs:plt_time[1]*fs,ch]),label= 'Raw Signal')
plt.plot((ampl_filt_signal[plt_time[0]*fs:plt_time[1]*fs]),label= 'High Gamma [80-125 Hz]')
plt.plot((phase_filt_signal[plt_time[0]*fs:plt_time[1]*fs]),label= 'Theta [4-8 Hz]')

plt.xlabel('Two Seconds of Theta Phase, High Gamma Amplitude, and raw signal')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plot_time_series(times, [sig, ampl_filt_signal, phase_filt_signal], ['Raw', 'Ampls', 'Phase'], xlim=[60, 62])

















