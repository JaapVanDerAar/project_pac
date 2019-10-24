import module_pac_functions as pacf
from scipy.signal import hilbert
import numpy as np
import matplotlib.pyplot as plt

#%%

def plot_signal(datastruct, phase_providing_band, amplitude_providing_band, subj, ch, fs): 
    
    """
    This function plots the raw signal and both phase and amplitude band
    
    Inputs: 
        Datastruct, the frequency bands, subject, channel, plot time window, and the fs
        HARDCODED: use same plot time window as is used for calculating PAC [2 3] in sec
    
    """
    
    # time over which pac is also calculated
    plt_time = [2, 3] 
    
    #calculating phase of theta
    phase_data = pacf.butter_bandpass_filter(datastruct[subj][ch], phase_providing_band[0], phase_providing_band[1], round(float(fs)));
    phase_data_hilbert = hilbert(phase_data);
#    phase_data_angle = np.angle(phase_data_hilbert);
    
    #calculating amplitude envelope of high gamma
    amp_data = pacf.butter_bandpass_filter(datastruct[subj][ch], amplitude_providing_band[0], amplitude_providing_band[1], round(float(fs)));
    amp_data_hilbert = hilbert(amp_data);
#    amp_data_abs = abs(amp_data_hilbert);
    
    plt.figure(figsize = (20,8));
    plt.plot((datastruct[subj][ch][plt_time[0]*fs:plt_time[1]*fs]),label= 'Raw Signal')
    plt.plot((amp_data_hilbert[plt_time[0]*fs:plt_time[1]*fs]),label= 'High Gamma [80-125 Hz]')
    plt.plot((phase_data_hilbert[plt_time[0]*fs:plt_time[1]*fs]),label= 'Theta [4-8 Hz]')
    
    plt.xlabel('subj: % 2d, ch: % 2d, Two Seconds of Theta Phase, High Gamma Amplitude, and raw signal' %(subj,ch))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
               
    