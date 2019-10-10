#setting settings and loading modules
from __future__ import division

%config InlineBackend.figure_format = 'retina'
%pylab inline

import time
import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy.io
import scipy.signal
from ipywidgets import interactive, IntSlider, FloatSlider, fixed
from IPython.display import display

#%%

f = 1024 #sampling frequency
dur = 10 #10 seconds of signal
freq = 7 #7 Hz signal
freq2 = 130 #130 Hz signal
t = arange(0, dur, 1/f) #times for d
sig1 = sin(2 * pi * freq * t) #10 Hz wavelength
sig1 = 1.5*sig1; #increase the power of signal 1
sig2 = sin(2 *pi * freq2 * t) #130 Hz wavelength
complex_signal = sig1+sig2;
plt.plot(t[0:512],complex_signal[0:512], label = 'complex signal') #plot 0.5 seconds of data
legend()

#%%

#Ignore the details of these functions for now
def butter_bandpass(lowcut, highcut, fs, order=4):
    #lowcut is the lower bound of the frequency that we want to isolate
    #hicut is the upper bound of the frequency that we want to isolate
    #fs is the sampling rate of our data
    nyq = 0.5 * fs #nyquist frequency - see http://www.dspguide.com/ if you want more info
    low = float(lowcut) / nyq
    high = float(highcut) / nyq
    b, a = sp.signal.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(mydata, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sp.signal.filtfilt(b, a, mydata)
    return y

#%%
#Using filters to isolate high and low frequencies
low_filtered_dat = butter_bandpass_filter(complex_signal,6,8,1024);
hi_filtered_dat = butter_bandpass_filter(complex_signal,120,140,1024);

figure(figsize = (15,6));
plt.plot(t[0:512],complex_signal[0:512], label = 'complex signal') #plot 0.5 seconds of data
plt.plot(t[0:512],low_filtered_dat[0:512], color = 'k', label = 'low frequency component') #plot 0.5 seconds of low frequency signal
plt.plot(t[0:512],hi_filtered_dat[0:512], color = 'g', label = 'high frequency component') #plot 0.5 seconds of high frequency signal
legend(loc='center left', bbox_to_anchor=(1, 0.5))

#%%
filename = 'emodat.mat'
filename = os.path.join('./', filename)
data = sp.io.loadmat(filename)
srate = data['srate'];
data = data['data']; # time series
data = data[0, :];
dat = data[0:1024];
def filter_dat(lo_cut,hi_cut,order):    

    filtdat = butter_bandpass_filter(dat,lo_cut,hi_cut,srate,order);
    fig = plt.figure(figsize=(24,6))
    ax1 = fig.add_subplot(2, 1, 1)
    plt.plot(dat,lw=2,color='blue',label = 'raw data')
    plt.plot(filtdat,lw=2,color='black',label='filtered data')
    plt.xlim(0, 1024)
    legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    return

a_slider = IntSlider(min=2, max=80, step=1, value=8)
b_slider = FloatSlider(min=2, max=150, step=1, value=12)
c_slider = FloatSlider(min=1, max=8, step=1, value=4)

w=interactive(filter_dat,lo_cut=a_slider,hi_cut=b_slider,order=c_slider)

display(w)

#%%

delta_funct = np.zeros(50);
delta_funct[25] = 1;


dat = data[0:1024];

#kernel 1: Gaussian Filter
x = np.linspace(0, 1, 10) #size of filter

mu = .5;
sig =.25;  ##
def generate_gaussian_filter(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)));

gaussian_filter = generate_gaussian_filter(x, mu, sig);

convolved_delt = sp.signal.convolve(delta_funct,gaussian_filter)
convolved_dat = sp.signal.convolve(dat,gaussian_filter)

fig = plt.figure(figsize=(24,6))
ax1 = fig.add_subplot(2, 1, 1);
ax1.plot(delta_funct,color = 'b',label = 'delta function')
ax1.plot(convolved_delt,color = 'r',label = 'filtered delta')
legend()
title('Filtering a Delta Function')


ax2 = fig.add_subplot(2, 1, 2);
ax2.plot(dat*2,color = 'b',label = 'raw data') #multiplied by 2 to scale to filtered data
ax2.plot(convolved_dat,color = 'r',label = 'filtered data')
legend()
title('Filtering a Delta Function')

#%%
dat = data[0:1024];

#time lag resistant kernel #1: Causal Filter (exponential filter)
#filter parameters
x = np.linspace(0, 1, 10) #size of filter
tau = .5;
amp_reduct = .1;
def generate_causal_filter(x,tau,amp_reduct):
    return list(reversed(amp_reduct*np.exp(x/tau)));

causal_filter = generate_causal_filter(x,tau,amp_reduct);
delta_funct = np.zeros((100));
delta_funct[50] = 1;

convolved_delt_causal_filt = sp.signal.convolve(delta_funct,causal_filter)
convolved_dat_causal_filt = sp.signal.convolve(dat,causal_filter)


#time lag resistant kernel #2: Two-way Filter
convolved_delt_two_way = sp.signal.convolve(delta_funct,gaussian_filter)
convolved_delt_two_way = convolved_delt_two_way[0:len(delta_funct)]
convolved_delt_two_way = sp.signal.convolve(list(reversed(convolved_delt_two_way)),gaussian_filter)
convolved_delt_two_way = list(reversed(convolved_delt_two_way[0:len(delta_funct)]))

convolved_dat_two_way = sp.signal.convolve(dat,gaussian_filter)
convolved_dat_two_way = convolved_dat_two_way[0:len(dat)]
convolved_dat_two_way = sp.signal.convolve(list(reversed(convolved_dat_two_way)),gaussian_filter)
convolved_dat_two_way = list(reversed(convolved_dat_two_way[0:len(dat)]))

#plotting
fig = plt.figure(figsize=(24,8))
ax1 = fig.add_subplot(2, 2, 1);
ax1.plot(delta_funct,color = 'b',label = 'delta function')
ax1.plot(convolved_delt_causal_filt,color = 'r',label = 'filtered delta')
legend()
title('Filtering a delta function with a causal filter')

ax2 = fig.add_subplot(2, 2, 2);
ax2.plot(dat*2,color = 'b',label = 'raw data') #multiplied by 2 to scale to filtered data
ax2.plot(convolved_dat_causal_filt,color = 'r',label = 'filtered data')
title('Filtering data with a causal filter')
legend()

ax3 = fig.add_subplot(2, 2, 3);
ax3.plot(delta_funct*10,color = 'b',label = 'delta function')
ax3.plot(convolved_delt_two_way,color = 'r',label = 'filtered delta')
legend()
title('Filtering a delta function with a two-way filter')

ax4 = fig.add_subplot(2, 2, 4);
ax4.plot(dat*50,color = 'b',label = 'raw data')
ax4.plot(convolved_dat_two_way, color = 'r', label = 'Filtering data with a two-way filter')
title('Filtering data with a two-way filter')
legend()