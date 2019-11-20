#%% Import standard libraries
import os
import numpy as np


#%% Import Voytek Lab libraries



#%% Change directory to import necessary modules

os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab\Code\distinguish_pac')

#%% import self-defined modules

import module_load_data as load_data
#import module_pac_functions as pacf
import module_detect_pac as detect_pac
import module_pac_plots as pac_plt


from bycycle.filt import lowpass_filter
from bycycle.features import compute_features

#%% Meta data

subjects = ['al','ca','cc','de','fp','gc','gf','gw',
          'h0','hh','jc','jm','jp','mv','rh','rr',
          'ug','wc','wm','zt']

os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab')

dat_name = 'fixation_pwrlaw'

#%% Parameters 

fs = 1000

timewindow = 100000/fs # 110800 is the length of shortest recording

#%% Run load data middle timewindow function
   
datastruct, elec_locs = load_data.load_data_timewindow(dat_name, subjects, fs, timewindow)


#%% Save structure

# np.save('datastruct_fpb', datastruct)

#%% Calculate largest peak in PSD using FOOF



from fooof import FOOOF
from neurodsp import spectral


epoch_len = 20 # in seconds
num_epochs = int(timewindow/epoch_len)

# initialze storing array
psd_peaks = []
backgr_params = []

for subj in range(len(datastruct)):
    
    # initialize channel specific storage array
    psd_peak_chs = []
    backgr_params_ch = []
    
    for ch in range(len(datastruct[subj])):
        
        
        # initialize epoch specific storage array
        psd_peak_ep = []
        backgr_params_ep = []
           
        for ep in range(num_epochs):
        
        
            
            # get signal
            sig = datastruct[subj][ch][(ep*fs*10):((ep+int((epoch_len/10)))*fs*10)]
            
            # compute frequency spectrum
            freq_mean, psd_mean = spectral.compute_spectrum(sig, fs, method='welch', avg_type='mean', nperseg=fs*2)
         
            # Set the frequency range upon which to fit FOOOF
            freq_range = [4, 55]
            bw_lims = [2, 8]
            max_n_peaks = 4
            
            if sum(psd_mean) == 0: 
                
                peak_params = np.empty([0, 3])
                
                psd_peak_ep.append(peak_params)
                
            else:
                
                # Initialize FOOOF model
                fm = FOOOF(peak_width_limits=bw_lims, background_mode='knee', max_n_peaks=max_n_peaks)
                
                # fit model
                fm.fit(freq_mean, psd_mean, freq_range) 
                
                # Central frequency, Amplitude, Bandwidth
                peak_params = fm.peak_params_
                
                #offset, knee, slope
                background_params = fm.background_params_
                
                if len(peak_params) > 0: 
                    
                    # find which peak has the biggest amplitude
                    max_ampl_idx = np.argmax(peak_params[:,1])
                        
                    # define biggest peak in power spectrum and add to channel array
                    psd_peak_ep.append(peak_params[max_ampl_idx])
                
                elif len(peak_params) == 0:
                    
                    psd_peak_ep.append(peak_params)
                
                backgr_params_ep.append(background_params)
                
        psd_peak_chs.append(psd_peak_ep)
        backgr_params_ch.append(backgr_params_ep)               
                
    psd_peaks.append(psd_peak_chs)
    backgr_params.append(backgr_params_ch)
    
    
#%% Calculate presence of PAC

amplitude_providing_band = [80, 125]; #80-125 Hz band

 # create output matrix of 20 * 64 (subj * max channels)
pac_presence = []
pac_pvals = []
pac_rhos = []

# for every subject
for subj in range(len(datastruct)):
        
    pac_presence_ch = []
    pac_pvals_ch = []
    pac_rhos_ch = []
    
    # for every channel
    for ch in range(len(datastruct[subj])):
        
        pac_presence_ep = np.full(num_epochs, np.nan)
        pac_pvals_ep = np.full(num_epochs, np.nan)
        pac_rhos_ep = np.full(num_epochs, np.nan)
        
        for ep in range(num_epochs):
        
            # for every channel that has peaks
            if len(psd_peaks[subj][ch][ep]) > 0:
            
                # define phase providing band
                CF = psd_peaks[subj][ch][ep][0]
                BW = psd_peaks[subj][ch][ep][2]
                
                phase_providing_band= [(CF - (BW/2)),  (CF + (BW/2))]
                
                data = datastruct[subj][ch][(ep*fs*10):((ep+int((epoch_len/10)))*fs*10)]
                
                #calculating phase of theta
                phase_data = pacf.butter_bandpass_filter(data, phase_providing_band[0], phase_providing_band[1], round(float(fs)));
                phase_data_hilbert = hilbert(phase_data);
                phase_data_angle = np.angle(phase_data_hilbert);
                
                #calculating amplitude envelope of high gamma
                amp_data = pacf.butter_bandpass_filter(data, amplitude_providing_band[0], amplitude_providing_band[1], round(float(fs)));
                amp_data_hilbert = hilbert(amp_data);
                amp_data_abs = abs(amp_data_hilbert);
                
                PAC_values = pacf.circle_corr(phase_data_angle, amp_data_abs)
            
                if PAC_values[1] <= 0.05:
                    
                    pac_presence_ep[ep] = 1
                    
                elif PAC_values[1] > 0.05: 
        
                    pac_presence_ep[ep] = 0
                    
                pac_pvals_ep[ep] = PAC_values[1]
                pac_rhos_ep[ep] = PAC_values[0]
        
        pac_presence_ch.append(pac_presence_ep)
        pac_pvals_ch.append(pac_pvals_ep)
        pac_rhos_ch.append(pac_rhos_ep)
        
    pac_presence.append(pac_presence_ch)
    pac_pvals.append(pac_pvals_ch)
    pac_rhos.append(pac_rhos_ch)
            
    print('another one is done =), this was subj', subj)
    
    
#%% Run resampling
 
num_resamples = 1000

resampled_rhovalues = []
resampled_pvalues = []

# for every subject
for subj in range(len(datastruct)):

    # create datastructs on channel level to save resamples PAC values
    resampled_pvalues_ch = []
    resampled_rhovalues_ch = []
    
    # for every channel
    for ch in range(len(datastruct[subj])):
            
        
        resampled_pvalues_ep = []
        resampled_rhovalues_ep = []
        
        for ep in range(num_epochs):
            
            
            # for every channel that has peaks
            if len(psd_peaks[subj][ch][ep]) > 0:
            
                # define phase providing band
                CF = psd_peaks[subj][ch][ep][0]
                BW = psd_peaks[subj][ch][ep][2]
                
                phase_providing_band = [(CF - (BW/2)),  (CF + (BW/2))]
            
                # get data of that channel
                data = datastruct[subj][ch][(ep*fs*10):((ep+int((epoch_len/10)))*fs*10)]
                
                # create array of random numbers between 0 and total samples 
                # which is as long as the number of resamples
                roll_array = np.random.randint(0, len(data), size=num_resamples)
                
                resampled_pvalues_sample = np.full(1000,np.nan)
                resampled_rhovalues_sample = np.full(1000,np.nan)
             
                # for every resample
                for ii in range(len(roll_array)):
        
                    # roll the phase data for a random amount 
                    phase_data_roll = np.roll(data, roll_array[ii])
                    
                    #calculating phase of theta
                    phase_data = pacf.butter_bandpass_filter(phase_data_roll, phase_providing_band[0], phase_providing_band[1], round(float(fs)));
                    phase_data_hilbert = hilbert(phase_data);
                    phase_data_angle = np.angle(phase_data_hilbert);
                    
                    #calculating amplitude envelope of high gamma
                    amp_data = pacf.butter_bandpass_filter(data, amplitude_providing_band[0], amplitude_providing_band[1], round(float(fs)));
                    amp_data_hilbert = hilbert(amp_data);
                    amp_data_abs = abs(amp_data_hilbert);
            
                    # calculate PAC
                    PAC_values = pacf.circle_corr(phase_data_angle, amp_data_abs)
                    
                    resampled_pvalues_sample[ii] = PAC_values[1]
                    resampled_rhovalues_sample[ii] = PAC_values[0]
                    
                resampled_pvalues_ep.append(resampled_pvalues_sample)
                resampled_rhovalues_ep.append(resampled_rhovalues_sample)
        
            
        resampled_pvalues_ch.append(resampled_pvalues_ep)
        resampled_rhovalues_ch.append(resampled_rhovalues_ep)                
        print('this was ch', ch)
        
    
    resampled_pvalues.append(resampled_pvalues_ch)
    resampled_rhovalues.append(resampled_rhovalues_ch)     
    
    print('another one is done =), this was subj', subj)

np.save('resampled_pvalues_20s', resampled_pvalues)
np.save('resampled_rhovalues_20s', resampled_rhovalues)
