#%%
#setting settings and loading modules
from __future__ import division

%config InlineBackend.figure_format = 'retina'
%pylab inline

import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy.io
import scipy.signal

#loading data
filename = 'emodat.mat'
filename = os.path.join('./', filename)

data = sp.io.loadmat(filename)
srate = data['srate']
data = data['data'] #ECoG data
data =  squeeze(data); #should be 1459920
print(srate) #should be 1017.25333333
print(len(data)) #should be 1476035 

#%% 

newsrate = 500 # new sampling rate

data_resampled = sp.signal.resample(data, int(np.floor(len(data)*(newsrate/srate))))

#How many seconds of data do we have?
print(str(len(data_resampled)) + ' total samples')
print(str(len(data_resampled)/newsrate) + ' seconds of data') #Total samples of data sampled at 1024 / sampling rate = total seconds sampled

#%%

f = 1024 #sampling frequency
dur = 10 #10 seconds of signal
freq = 7 #7 Hz signal
freq2 = 130 #130 Hz signal
t = arange(0, dur, 1/f) #times for d
sig1 = sin(2 * pi * freq * t) #10 Hz wavelength
sig1 = 1.5*sig1; #increase the power of signal 1
sig2 = sin(2 *pi * freq2 * t) #130 Hz wavelength
figure(figsize=(16,6))
plt.plot(t[0:512],sig1[0:512]+sig2[0:512], label = 'complex signal') #plot 0.5 seconds of data
legend()

#%% 

# calculating fourier transform of complex signal
fourier = np.fft.fft(sig1+sig2)

# finding frequency values for the x axis
fx_step_size = f/len(sig1)
nyq = .5*newsrate
total_steps = nyq/fx_step_size
fx_bins = np.linspace(0,nyq,total_steps)

# plotting up to 200 Hz
plot(fx_bins[0:2500],abs(fourier[0:2500]))
plt.ylabel('Power')
plt.xlabel('Frequency (Hz)')
plt.title('FFT of a complex signal')

#%%

plot(np.arange(0, 1024)/1024., data[10000:11024])
plt.ylabel('Voltage (uV)')
plt.xlabel('Time [s]')
plt.title('ECoG Signal in Time Domain')

#%%

# calculating fourier transform of complex signal
# we're going to take a sample of the data to keep fx bins at a reasonable size.
fourier = np.fft.fft(data_resampled[0:10000]) 

# finding frequency values for the x axis
fx_step_size = newsrate/len(data_resampled[0:10000])
nyq = .5*newsrate
total_steps = nyq/fx_step_size
fx_bins = np.linspace(0,nyq,total_steps)

figure(figsize=(16,10))
plt.subplot(1,2,1)
plot(fx_bins[0:4000],abs(fourier[0:4000])) #any frequencies above 200 Hz are probably noise
plt.ylabel('Power')
plt.xlabel('Frequency (Hz)')
plt.title('FFT of ECoG signal')

plt.subplot(1,2,2)
plot(fx_bins[0:4000],log(abs(fourier[0:4000]))) #any frequencies above 200 Hz are probably noise
plt.ylabel('log Power')
plt.xlabel('Frequency (Hz)')
plt.title('FFT of ECoG signal')

#%%

f,pspec = sp.signal.welch(data, fs=newsrate, window='hanning', nperseg=10
                          *newsrate, noverlap=newsrate/2, nfft=None, detrend='linear', return_onesided=True, scaling='density')
#Try to figure out what the paramaters above are doing. What happens if you change them?

figure(figsize=(16,10))
loglog(f[0:200*2],pspec[0:200*2])
#Any frequencies with >200 Hz are going to be noise in ECoG data.
plt.ylabel('Power')
plt.xlabel('Frequency (Hz)')
plt.xlim([1, 150])
plt.title("Welch's PSD of ECoG signal")

