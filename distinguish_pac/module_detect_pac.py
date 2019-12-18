import pandas as pd
import module_pac_functions as pacf
from scipy.signal import hilbert
import numpy as np
import scipy.io
from fooof import FOOOF
from neurodsp import spectral

import time


#%%  Calculate PAC

def cal_pac_values(datastruct, phase_providing_band, amplitude_providing_band, fs):
    """ iterates over all subjects and channels, calculates the PAC and results in dataframe
    HARDCODED: calculates the PAC of 1 second ([2 3]second range in data)
    Inputs:
    -   Datastruct [subj * channels * data ] in numpy arrays
    -   Phase and amplitude frequency bands
    -   Sampling Frequency"""    
    
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
            phase_data = pacf.butter_bandpass_filter(data, phase_providing_band[0], phase_providing_band[1], round(float(fs)));
            phase_data_hilbert = hilbert(phase_data);
            phase_data_angle = np.angle(phase_data_hilbert);
            
            #calculating amplitude envelope of high gamma
            amp_data = pacf.butter_bandpass_filter(data, amplitude_providing_band[0], amplitude_providing_band[1], round(float(fs)));
            amp_data_hilbert = hilbert(amp_data);
            amp_data_abs = abs(amp_data_hilbert);
            
            
            PAC_values = pacf.circle_corr(phase_data_angle[2000:3000], amp_data_abs[2000:3000])
        
            if PAC_values[1] <= 0.05:
                
                pac_presence[ch][subj] = 1
                
            elif PAC_values[1] > 0.05: 
    
                pac_presence[ch][subj] = 0
                
            pac_pvals[ch][subj] = PAC_values[1]
            pac_rhos[ch][subj] = PAC_values[0]
                
        print('another one is done =), this was subj', subj)
    
    return pac_presence, pac_pvals, pac_rhos


#%% Calculate resampled Rho values
    
def resampled_pac(datastruct, phase_providing_band, amplitude_providing_band, fs, num_resamples):
    """
    This function calculated the 'true' PAC by resampling data 1000 times, 
    calculating the rho values for every resample, whereafter the true p-value 
    is calculated
    
    Inputs:
        Datastructure, phase and amplitude band, fs and number of resamples
        
    Outputs:
        Resampled rho values
        Resamples p values
    
    """
    resampled_rhovalues_subjlevel = []
    resampled_pvalues_subjlevel = []
    
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
            roll_array = np.random.randint(0, len(data), size=num_resamples)
            
            resampled_pvalues_sample = []
            resampled_rhovalues_sample = []
         
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
                PAC_values = pacf.circle_corr(phase_data_angle[2000:3000], amp_data_abs[2000:3000])
                
                resampled_pvalues_sample.append(PAC_values[1])
                resampled_rhovalues_sample.append(PAC_values[0])
                
            resampled_pvalues_channel.append(resampled_pvalues_sample)
            resampled_rhovalues_channel.append(resampled_rhovalues_sample)
       
            print('this was ch', ch)
        
        
        resampled_pvalues_subjlevel.append(resampled_pvalues_channel)
        resampled_rhovalues_subjlevel.append(resampled_rhovalues_channel)     
        
        print('another one is done =), this was subj', subj)
    
    return resampled_rhovalues_subjlevel, resampled_pvalues_subjlevel
        
#%% Calculate true values after resampling        
   
def true_pac_values(pac_rhos, resamp_rho):
    """
    
    This function calculated the true p values based on the statistical resampling
    of the resampled_pac function
    
    Inputs:
        Measured Rho value
        Resampled Rho values
        
    Outputs: 
        True PAC zvalue
        True PAC pvalue
        Binary presence of PAC matrix
    
    """
    
    pac_true_zvals = pd.DataFrame(np.nan, index=range(len(resamp_rho)), columns=range(64))
    pac_true_pvals = pd.DataFrame(np.nan, index=range(len(resamp_rho)), columns=range(64))
    pac_true_presence = pd.DataFrame(np.nan, index=range(len(resamp_rho)), columns=range(64))
    
    
    # for every subj
    for subj in range(len(resamp_rho)):
        
        # for every channel
        for ch in range(len(resamp_rho[subj])):
            
            true_z = (pac_rhos[subj][ch] - np.mean(resamp_rho[subj][ch])) / np.std(resamp_rho[subj][ch])
            p_value = scipy.stats.norm.sf(abs(true_z))
                
            pac_true_pvals[ch][subj] = p_value
            pac_true_zvals[ch][subj] = true_z
            
            if pac_true_zvals[ch][subj] >= 0  and  pac_true_pvals[ch][subj] <= 0.05:
    
                pac_true_presence[ch][subj] = 1         
                
            else: 
            
                pac_true_presence[ch][subj] = 0 
                
    return pac_true_zvals, pac_true_pvals, pac_true_presence
       
#%%  Calculate PAC - but phase providing band is variabel

def cal_pac_values_varphase(datastruct, amplitude_providing_band, fs, psd_peaks):
    """ iterates over all subjects and channels, calculates the PAC and results in dataframe
    Same function as cal_pac_values but:
        with a variable phase providing band
        and for the full timeframe instead of 1 second
        
    Inputs:
    -   Datastruct [subj * channels * data ] in numpy arrays
    -   Amplitude frequency band
    -   PSD peak array with CF and BW
    -   Sampling Frequency"""    
    
    # create output matrix of 20 * 64 (subj * max channels)
    pac_presence = pd.DataFrame(np.nan, index=range(len(datastruct)), columns=range(64))
    pac_pvals = pd.DataFrame(np.nan, index=range(len(datastruct)), columns=range(64))
    pac_rhos = pd.DataFrame(np.nan, index=range(len(datastruct)), columns=range(64))

    # for every subject
    for subj in range(len(datastruct)):
        
        # for every channel
        for ch in range(len(datastruct[subj])):
            
            # for every channel that has peaks
            if len(psd_peaks[subj][ch]) > 0:
            
                # define phase providing band
                CF = psd_peaks[subj][ch][0]
                BW = psd_peaks[subj][ch][2]
                
                phase_providing_band= [(CF - (BW/2)),  (CF + (BW/2))]
                
                data = datastruct[subj][ch] 
                
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
                    
                    pac_presence[ch][subj] = 1
                    
                elif PAC_values[1] > 0.05: 
        
                    pac_presence[ch][subj] = 0
                    
                pac_pvals[ch][subj] = PAC_values[1]
                pac_rhos[ch][subj] = PAC_values[0]
                
        print('another one is done =), this was subj', subj)
    
    return pac_presence, pac_pvals, pac_rhos

#%% Calculate resampled Rho values - with phase frequency variable
    
def resampled_pac_varphase(datastruct, amplitude_providing_band, fs, num_resamples, psd_peaks):
    """
    This function calculated the 'true' PAC by resampling data 1000 times, 
    calculating the rho values for every resample, whereafter the true p-value 
    is calculated
    Same function as resampled_pac but:
        with a variable phase providing band
        and for the full timeframe instead of 1 second
    
    Inputs:
        Datastructure, phase and amplitude band, fs and number of resamples
        
    Outputs:
        Resampled rho values
        Resamples p values
    
    """
    resampled_rhovalues_subjlevel = []
    resampled_pvalues_subjlevel = []
    
    # for every subject
    for subj in range(len(datastruct)):
    
        # create datastructs on channel level to save resamples PAC values
        resampled_pvalues_channel = []
        resampled_rhovalues_channel = []
        
        # for every channel
        for ch in range(len(datastruct[subj])):
            
            # time it       
            start = time.time()
                
           
            # for every channel that has peaks
            if len(psd_peaks[subj][ch]) > 0:
            
                # define phase providing band
                CF = psd_peaks[subj][ch][0]
                BW = psd_peaks[subj][ch][2]
                
                phase_providing_band = [(CF - (BW/2)),  (CF + (BW/2))]
            
                # get data of that channel
                data = datastruct[subj][ch] 
                
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
                    
                resampled_pvalues_channel.append(resampled_pvalues_sample)
                resampled_rhovalues_channel.append(resampled_rhovalues_sample)
           
                print('this was ch', ch)
            
                end = time.time()
                print(end - start)   
        
        resampled_pvalues_subjlevel.append(resampled_pvalues_channel)
        resampled_rhovalues_subjlevel.append(resampled_rhovalues_channel)     
        
        print('another one is done =), this was subj', subj)
    
    return resampled_rhovalues_subjlevel, resampled_pvalues_subjlevel
        
#%% Using FOOOF, select the biggest peak in the periodic power spectrum model
    

def fooof_highest_peak(datastruct, fs, freq_range, bw_lims, max_n_peaks, freq_range_long):
    """
    This function is build on FOOOF and neuroDSP. It fits a model to the power 
    frequency spectrum, look for the biggest peak (in amplitude) and extracts 
    the characteristics of the peak
    
    Inputs: 
        datastruct and fs
        
        FOOOF parameters:
        Frequency Range
        Min and Max BW
        Max # of Peaks  
        Long frequency range
        
    Outputs:
        Arrays of biggest peak characterics [CF, Ampl, BW]
        Array of background parameters [Exp, knee, offset]
        Array of background parameters long [Exp, knee, offset]
    
    """
    
    
    # initialze storing array
    psd_peaks = []
    backgr_params = []
    backgr_params_long = []
    
    for subj in range(len(datastruct)):
        
        # initialize channel specific storage array
        psd_peak_chs = []
        backgr_params_ch = []
        backgr_params_long_ch = []
        
        for ch in range(len(datastruct[subj])):
            
            # get signal
            sig = datastruct[subj][ch]
            
            # compute frequency spectrum
            freq_mean, psd_mean = spectral.compute_spectrum(sig, fs, method='welch', avg_type='mean', nperseg=fs*2)
            
            # if dead channel
            if sum(psd_mean) == 0: 
                
                peak_params = np.array([np.nan, np.nan, np.nan])
                background_params = np.array([np.nan, np.nan, np.nan])
                background_params_long = np.array([np.nan, np.nan, np.nan])
                
                psd_peak_chs.append(peak_params)
                backgr_params_ch.append(background_params)
                backgr_params_long_ch.append(background_params_long)
                
            else:
                
                # Initialize FOOOF model
                fm = FOOOF(peak_width_limits=bw_lims, background_mode='knee', max_n_peaks=max_n_peaks)
                
                # fit model
                fm.fit(freq_mean, psd_mean, freq_range) 
                
                # Central frequency, Amplitude, Bandwidth
                peak_params = fm.peak_params_
                
                #offset, knee, slope
                background_params = fm.background_params_
                    
                # if peaks are found
                if len(peak_params) > 0: 
                    
                    # find which peak has the biggest amplitude
                    max_ampl_idx = np.argmax(peak_params[:,1])
                    
                    # find this peak hase the following characteristics:
                    # 1) CF under 15 Hz
                    # 2) Amp above .2
                    # 3) Amp under 1.5 (to get rid of artifact)
                    if ((peak_params[max_ampl_idx][0] < 15) &  \
                        (peak_params[max_ampl_idx][1] >.2) &  \
                        (peak_params[max_ampl_idx][1] < 1.5)):
                        
                        # write it to channel array
                        psd_peak_chs.append(peak_params[max_ampl_idx])
                      
                    # otherwise write empty
                    else:   
                        peak_params = np.array([np.nan, np.nan, np.nan])
                        psd_peak_chs.append(peak_params)
                        
                # if no peaks are found, write empty
                elif len(peak_params) == 0:
                    
                    # write empty
                    peak_params = np.array([np.nan, np.nan, np.nan])
                    psd_peak_chs.append(peak_params)
                
                # add backgr parameters
                backgr_params_ch.append(background_params)
                
                
                # get the long frequency range aperiodic parameters 
                # Initialize FOOOF model
                fm = FOOOF(peak_width_limits=bw_lims, background_mode='knee', max_n_peaks=max_n_peaks)
                
                # fit model with long range
                fm.fit(freq_mean, psd_mean, freq_range_long) 
                
                #offset, knee, slope of long range
                background_params_long = fm.background_params_
                
                # add long background parameters
                backgr_params_long_ch.append(background_params_long)
                      
                    
        psd_peaks.append(psd_peak_chs)
        backgr_params.append(backgr_params_ch)
        backgr_params_long.append(backgr_params_long_ch)
        
            
    return psd_peaks, backgr_params, backgr_params_long

