

#%%  EXAMPLE 1:  Sawtooth with changing sharp edge.

import os
os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab')


# imports
from scipy import signal
import numpy as np
from scipy.signal import hilbert
from neurodsp import spectral
from fooof import FOOOF
from neurodsp.plts.spectral import plot_power_spectra
import matplotlib.pyplot as plt

os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab\Code\distinguish_pac')
import module_pac_functions as pacf
import module_detect_pac as detect_pac


#%% Create sawtooth shaped data

dt = 0.001;    #Sampling interval = 1 ms.
sig = []
for ii in range(0,1000):
    
    duration = np.ceil(166 + np.random.normal()*15);
    r = 0.90 + 0.1*np.random.rand();    
    a = list(range(0,int(duration)))/duration
    sig.extend(signal.sawtooth(a, r));

 
sig = sig + 0.02*np.random.normal(0, 1, np.shape(sig));
sOrig = sig;
sig = sig[0:12000];
sCropped = sig[1000:len(sig)-1000];


plt_time = [0, 5000]
fs = 1000

fig = plt.figure(figsize=(20,8))
plt.plot(sig[plt_time[0]:plt_time[1]])

#%%
# 3 polycoherence
from polycoherence import _plot_signal, polycoherence, plot_polycoherence
from math import pi
from scipy.fftpack import next_fast_len
 
# plot biocoherence 
#sig = sig[plt_time[0]:plt_time[1]]
nn = len(sig)
t = np.linspace(0, 100, nn)
kw = dict(nperseg=nn // 10, noverlap=nn // 20, nfft=next_fast_len(nn // 2))

freq1, freq2, bicoh = polycoherence(sig, fs, flim1=(40, 80), flim2=(3, 18), **kw)
plot_polycoherence(freq1, freq2, bicoh)
plt.show()
#%% Get amplitude and phase data
phase_providing_band = [4,8]
amplitude_providing_band = [40,100]

#calculating phase of theta
phase_data = pacf.butter_bandpass_filter(sig, phase_providing_band[0], phase_providing_band[1], round(float(fs)));
phase_data_hilbert = hilbert(phase_data);
phase_data_angle = np.angle(phase_data_hilbert);

#calculating amplitude envelope of high gamma
amp_data = pacf.butter_bandpass_filter(sig, amplitude_providing_band[0], amplitude_providing_band[1], round(float(fs)));
amp_data_hilbert = hilbert(amp_data);
amp_data_abs = abs(amp_data_hilbert);

#%% Plot

plt.figure(figsize = (20,8));
plt.plot((sig[plt_time[0]:plt_time[1]]+.2),label= 'Raw Signal')
plt.plot((amp_data_hilbert[plt_time[0]:plt_time[1]]),label= 'High Gamma [80-125 Hz]')
plt.plot((phase_data_hilbert[plt_time[0]:plt_time[1]]),label= 'Phase [{0:.2f} - {1:.2f} Hz]'.format(phase_providing_band[0], phase_providing_band[1]))
plt.plot((amp_data_abs[plt_time[0]:plt_time[1]]),label= 'Amplitude Envelope')
plt.plot((phase_data_angle[plt_time[0]:plt_time[1]]*0.1),label= 'Phase Angle')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))


#%% 

freq_mean, psd_mean = spectral.compute_spectrum(sig, fs, method='welch', avg_type='mean', nperseg=fs*2)
plot_power_spectra(freq_mean[:200], psd_mean[:200], 'Welch')

freq_range = [2,80] # this is the range gao uses
bw_lims = [2, 8]
max_n_peaks =4


# Initialize FOOOF model
fm = FOOOF(peak_width_limits=bw_lims, background_mode='knee', max_n_peaks=max_n_peaks)

# fit model
fm.fit(freq_mean, psd_mean, freq_range) 

fm.report()


#%% Create Imperfect sinusoids  

plt_time = [0, 1000]
fs = 1000

f =12.0;
dt = 0.001;
sig = [];

for hello in range(0,400): 
    
    fi = 50.0;
    fii = 10.0;
    
        
    x1 = -np.cos(np.pi* np.linspace(0,1,int(1/(dt*f))+1))
    x2 =  np.cos(np.pi* np.linspace(0,1,int(1/(dt*f))+1))
    
    theta = np.concatenate((x1, x2), axis=None);   
    
    
    r = np.ceil(np.random.rand()*5);                   #Define the duration of the sharp edge.
    x3i = np.linspace(0,1,int(r+1));                   #Create the sharp edge.
   
  
    x3ii = (np.cos(np.pi * np.linspace(0,1,int(1/(dt*fii))+1)) + 1) / 2 
    # x3ii = np.cos(np.pi *  np.linspace(0,1,int(1/(dt*fii))+1)+1)/2 #Define the taper of the sharp edge.
    x3 = np.concatenate((x3i, x3ii), axis=None);       #Create the tapered sharp edge.

    pos = np.ceil(np.random.rand()*10)+15;             #Insert the sharp edge into the sinusoid.
    seed1 = theta[:]
    seed1[int(pos)-1:int(pos+len(x3))-1] = seed1[int(pos)-1:int(pos+len(x3))-1]+x3;

    sig.extend(seed1)


sig = sig + 0.1*np.random.normal(-2, 2, np.shape(sig));
    
# Calculate bands   
    
phase_providing_band = [4,8]
amplitude_providing_band = [40,100]

#calculating phase of theta
phase_data = pacf.butter_bandpass_filter(sig, phase_providing_band[0], phase_providing_band[1], round(float(fs)));
phase_data_hilbert = hilbert(phase_data);
phase_data_angle = np.angle(phase_data_hilbert);

#calculating amplitude envelope of high gamma
amp_data = pacf.butter_bandpass_filter(sig, amplitude_providing_band[0], amplitude_providing_band[1], round(float(fs)));
amp_data_hilbert = hilbert(amp_data);
amp_data_abs = abs(amp_data_hilbert);   
    
    
  
# Plot

plt.figure(figsize = (20,8));
plt.plot((sig[plt_time[0]:plt_time[1]]),label= 'Raw Signal')
plt.plot((amp_data_hilbert[plt_time[0]:plt_time[1]]),label= 'High Gamma [80-125 Hz]')
plt.plot((phase_data_hilbert[plt_time[0]:plt_time[1]]),label= 'Phase [{0:.2f} - {1:.2f} Hz]'.format(phase_providing_band[0], phase_providing_band[1]))
plt.plot((amp_data_abs[plt_time[0]:plt_time[1]]),label= 'Amplitude Envelope')
plt.plot((phase_data_angle[plt_time[0]:plt_time[1]]),label= 'Phase Angle')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title('Stimulated data with imperfect sinusoide data give rise to PAC')

#%% 

freq_mean, psd_mean = spectral.compute_spectrum(sig, fs, method='welch', avg_type='mean', nperseg=fs*2)
plot_power_spectra(freq_mean[:200], psd_mean[:200], 'Welch')

freq_range = [2,80] # this is the range gao uses
bw_lims = [2, 8]
max_n_peaks =4


# Initialize FOOOF model
fm = FOOOF(peak_width_limits=bw_lims, background_mode='knee', max_n_peaks=max_n_peaks)

# fit model
fm.fit(freq_mean, psd_mean, freq_range) 

fm.report()  
    
#%% Run these data cycle by cycle 
from bycycle.filt import lowpass_filter
from bycycle.features import compute_features


fs = 1000
f_range = [4, 8]
f_lowpass = 55
N_seconds = len(sig) / fs - 2

signal = lowpass_filter(sig, fs, f_lowpass, N_seconds=N_seconds, remove_edge_artifacts=False)

plt.plot(sig[0:1500])
plt.show()
df = compute_features(signal, fs, f_range)
 
plt.scatter(df.time_ptsym, df.time_rdsym)

