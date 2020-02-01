import pandas as pd
import module_pac_functions as pacf
from scipy.signal import hilbert
import numpy as np
import scipy.io
from fooof import FOOOF
from neurodsp import spectral
from module_load_data import get_signal
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

def cal_pac_values_varphase(datastruct, amplitude_providing_band, fs, features_df, epoch_len):
    """ iterates over all subjects and channels, calculates the PAC and results in dataframe
    Same function as cal_pac_values but:
        with a variable phase providing band
        and for the full timeframe instead of 1 second
        
    Inputs:
    -   Datastruct [subj * channels * data ] in numpy arrays
    -   Amplitude frequency band
    -   Sampling Frequency  
    -   Features df
    
    """  
 
    # create output columns
    features_df['pac_presence']  = np.int64
    features_df['pac_pvals'] = np.nan
    features_df['pac_rhos']  = np.nan

    for ii in range(len(features_df)):
            
        # for every channel that has peaks
        if ~np.isnan(features_df['CF'][ii]):
        
            # define phase providing band
            CF = features_df['CF'][ii]
            BW = features_df['BW'][ii]
            
            phase_providing_band= [(CF - (BW/2)),  (CF + (BW/2))]
            
            subj = features_df['subj'][ii]
            ch = features_df['ch'][ii]
            ep = features_df['ep'][ii]
            data = datastruct[subj][ch][(ep*fs*epoch_len):((ep*fs*epoch_len)+fs*epoch_len)]
            
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
                
                features_df['pac_presence'][ii] = 1
                
            elif PAC_values[1] > 0.05: 
    
                features_df['pac_presence'][ii] = 0
                
            features_df['pac_pvals'][ii] = PAC_values[1]
            features_df['pac_rhos'][ii] = PAC_values[0]
    
    return features_df

#%%  Calculate PAC for monkey data - but phase providing band is variabel

def cal_pac_values_varphase_monkey(datastruct, amplitude_providing_band, fs, features_df):
    """ iterates over all subjects and channels, calculates the PAC and results in dataframe
    Same function as cal_pac_values but:
        with a variable phase providing band
        and for the full timeframe instead of 1 second
        
    Inputs:
    -   Datastruct [subj * channels * data ] in numpy arrays
    -   Amplitude frequency band
    -   Sampling Frequency  
    -   Features df
    
    """  
 
    # create output columns
    features_df['pac_presence']  = np.int64
    features_df['pac_pvals'] = np.nan
    features_df['pac_rhos']  = np.nan

    for ii in range(len(features_df)):
            
        # for every channel that has peaks
        if ~np.isnan(features_df['CF'][ii]):
        
            # define phase providing band
            CF = features_df['CF'][ii]
            BW = features_df['BW'][ii]
            
            phase_providing_band= [(CF - (BW/2)),  (CF + (BW/2))]
            
            subj = features_df['subj'][ii]
            ch = features_df['ch'][ii]
            ep = features_df['ep'][ii]
            data = datastruct[subj][ch][ep]
            
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
                
                features_df['pac_presence'][ii] = 1
                
            elif PAC_values[1] > 0.05: 
    
                features_df['pac_presence'][ii] = 0
                
            features_df['pac_pvals'][ii] = PAC_values[1]
            features_df['pac_rhos'][ii] = PAC_values[0]
            
            print('this was ch', ii)
    
    return features_df

#%% Calculate resampled Rho values - with phase frequency variable
    
def resampled_pac_varphase(datastruct, amplitude_providing_band, fs, num_resamples, features_df, epoch_len):
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
    # create output columns
    features_df['resamp_pac_presence']  = np.int64
    features_df['resamp_pac_pvals'] = np.nan
    features_df['resamp_pac_zvals']  = np.nan
    
    for ii in range(len(features_df)):
        
        # time it       
        start = time.time()
            
       
        # for every channel that has periodic peak
        if ~np.isnan(features_df['CF'][ii]):
        
            # define phase providing band
            CF = features_df['CF'][ii]
            BW = features_df['BW'][ii]
            
            phase_providing_band= [(CF - (BW/2)),  (CF + (BW/2))]
            
            # get data
            subj = features_df['subj'][ii]
            ch = features_df['ch'][ii]
            ep = features_df['ep'][ii]
            data = datastruct[subj][ch][(ep*fs*epoch_len):((ep*fs*epoch_len)+fs*epoch_len)]

            
            # create array of random numbers between 0 and total samples 
            # which is as long as the number of resamples
            roll_array = np.random.randint(0, len(data), size=num_resamples)
            
            resampled_pvals = np.full(num_resamples,np.nan)
            resampled_rhos = np.full(num_resamples,np.nan)
         
            # for every resample
            for jj in range(len(roll_array)):
    
                # roll the phase data for a random amount 
                phase_data_roll = np.roll(data, roll_array[jj])
                
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
                
                resampled_pvals[jj] = PAC_values[1]
                resampled_rhos[jj] = PAC_values[0]
            
            # after rolling data and calculating 1000 rho and pac's 
            # calculate the true values after resampling
            true_z = (features_df['pac_rhos'][ii] - np.mean(resampled_rhos)) / np.std(resampled_rhos)
            p_value = scipy.stats.norm.sf(abs(true_z))
         
            # write to dataframe
            features_df['resamp_pac_pvals'][ii] = p_value
            features_df['resamp_pac_zvals'][ii] = true_z
            
            # is it significant?
            if true_z >= 0  and  p_value <= 0.05:
    
                features_df['resamp_pac_presence'][ii] = 1         
                
            else: 
            
                features_df['resamp_pac_presence'][ii] = 0 
            
       
            print('this was ch', ii)
        
            end = time.time()
            print(end - start)   
        
    return features_df
        

#%% Calculate resampled Rho values - with phase frequency variable
    
def resampled_pac_varphase_monkey(datastruct, amplitude_providing_band, fs, num_resamples, features_df):
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
    # create output columns
    features_df['resamp_pac_presence']  = np.int64
    features_df['resamp_pac_pvals'] = np.nan
    features_df['resamp_pac_zvals']  = np.nan
    
    for ii in range(len(features_df)):
        
        # time it       
        start = time.time()
            
       
        # for every channel that has periodic peak
        if ~np.isnan(features_df['CF'][ii]):
        
            # define phase providing band
            CF = features_df['CF'][ii]
            BW = features_df['BW'][ii]
            
            phase_providing_band= [(CF - (BW/2)),  (CF + (BW/2))]
            
            # get data
            subj = features_df['subj'][ii]
            ch = features_df['ch'][ii]
            ep = features_df['ep'][ii]
            data = datastruct[subj][ch][ep]

            
            # create array of random numbers between 0 and total samples 
            # which is as long as the number of resamples
            roll_array = np.random.randint(0, len(data), size=num_resamples)
            
            resampled_pvals = np.full(num_resamples,np.nan)
            resampled_rhos = np.full(num_resamples,np.nan)
         
            # for every resample
            for jj in range(len(roll_array)):
    
                # roll the phase data for a random amount 
                phase_data_roll = np.roll(data, roll_array[jj])
                
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
                
                resampled_pvals[jj] = PAC_values[1]
                resampled_rhos[jj] = PAC_values[0]
            
            # after rolling data and calculating 1000 rho and pac's 
            # calculate the true values after resampling
            true_z = (features_df['pac_rhos'][ii] - np.mean(resampled_rhos)) / np.std(resampled_rhos)
            p_value = scipy.stats.norm.sf(abs(true_z))
         
            # write to dataframe
            features_df['resamp_pac_pvals'][ii] = p_value
            features_df['resamp_pac_zvals'][ii] = true_z
            
            # is it significant?
            if true_z >= 0  and  p_value <= 0.05:
    
                features_df['resamp_pac_presence'][ii] = 1         
                
            else: 
            
                features_df['resamp_pac_presence'][ii] = 0 
            
       
            print('this was ch', ii)
        
            end = time.time()
            print(end - start)   
        
    return features_df

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
                    # 1) CF under 30 Hz
                    # 2) Amp above .2
                    # 3) Amp under 1.5 (to get rid of artifact)
                    if ((peak_params[max_ampl_idx][0] < 30) &  \
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

#%% Using FOOOF, select the biggest peak in the periodic power spectrum model with epochs
    

def fooof_highest_peak_epoch(datastruct, epoch_len, fs, freq_range, bw_lims, max_n_peaks, freq_range_long):
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
    
    num_epochs = int(len(datastruct[0][0]) / fs / epoch_len)
    
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
            
            # initialize channel specific storage array
            psd_peak_ep = []
            backgr_params_ep = []
            backgr_params_long_ep = [] 
            
            for ep in range(num_epochs):
            
                # get signal
                sig = datastruct[subj][ch][(ep*fs*epoch_len):((ep*fs*epoch_len)+fs*epoch_len)]    
                
                # compute frequency spectrum
                freq_mean, psd_mean = spectral.compute_spectrum(sig, fs, method='welch', avg_type='mean', nperseg=fs*2)
                
                # if dead channel
                if sum(psd_mean) == 0: 
                    
                    peak_params = np.array([np.nan, np.nan, np.nan])
                    background_params = np.array([np.nan, np.nan, np.nan])
                    background_params_long = np.array([np.nan, np.nan, np.nan])
                    
                    psd_peak_ep.append(peak_params)
                    backgr_params_ep.append(background_params)
                    backgr_params_long_ep.append(background_params_long)
                    
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
                        # 1) CF under 30 Hz
                        # 2) Amp above .2
                        # 3) Amp under 1.5 (to get rid of artifact)
                        if ((peak_params[max_ampl_idx][0] < 30) &  \
                            (peak_params[max_ampl_idx][1] >.2) &  \
                            (peak_params[max_ampl_idx][1] < 1.5)):
                            
                            # write it to channel array
                            psd_peak_ep.append(peak_params[max_ampl_idx])
                          
                        # otherwise write empty
                        else:   
                            peak_params = np.array([np.nan, np.nan, np.nan])
                            psd_peak_ep.append(peak_params)
                            
                    # if no peaks are found, write empty
                    elif len(peak_params) == 0:
                        
                        # write empty
                        peak_params = np.array([np.nan, np.nan, np.nan])
                        psd_peak_ep.append(peak_params)
                    
                    # add backgr parameters
                    backgr_params_ep.append(background_params)
                    
                    
                    # get the long frequency range aperiodic parameters 
                    # Initialize FOOOF model
                    fm = FOOOF(peak_width_limits=bw_lims, background_mode='knee', max_n_peaks=max_n_peaks)
                    
                    # fit model with long range
                    fm.fit(freq_mean, psd_mean, freq_range_long) 
                    
                    #offset, knee, slope of long range
                    background_params_long = fm.background_params_
                    
                    # add long background parameters
                    backgr_params_long_ep.append(background_params_long)
                   
            # write to arrays on ch level       
            psd_peak_chs.append(psd_peak_ep)
            backgr_params_ch.append(backgr_params_ep)      
            backgr_params_long_ch.append(backgr_params_long_ep)
        
        # write to arrays on subj level
        psd_peaks.append(psd_peak_chs)
        backgr_params.append(backgr_params_ch)
        backgr_params_long.append(backgr_params_long_ch)
        
            
    return psd_peaks, backgr_params, backgr_params_long


#%% 
    

def fooof_highest_peak_epoch_4_12(datastruct, epoch_len, fs, freq_range, bw_lims, max_n_peaks, freq_range_long, features_df):
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
        
    # initialize space in features_df     
    # periodic component
    features_df['CF'] = np.nan
    features_df['Amp'] = np.nan
    features_df['BW'] = np.nan
    
    # aperiodic component
    features_df['offset'] = np.nan
    features_df['knee'] = np.nan
    features_df['exp'] = np.nan
    
    # aperiodic component long frequency range
    features_df['offset_long'] = np.nan
    features_df['knee_long'] = np.nan
    features_df['exp_long'] = np.nan
    
    for ii in range(len(features_df)):
  
        # get data
        subj = features_df['subj'][ii]
        ch = features_df['ch'][ii]
        ep = features_df['ep'][ii]
        sig = datastruct[subj][ch][(ep*fs*epoch_len):((ep*fs*epoch_len)+fs*epoch_len)]
                
        # compute frequency spectrum
        freq_mean, psd_mean = spectral.compute_spectrum(sig, fs, method='welch', avg_type='mean', nperseg=fs*2)
        
        # if dead channel
        if sum(psd_mean) == 0: 
            
            # no oscillation was found
            features_df['CF'][ii] = np.nan
            features_df['Amp'][ii] = np.nan
            features_df['BW'][ii] = np.nan
            
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
                # 1) CF between 4 and 12 Hz
                # 2) Amp above .2
                # 3) Amp under 1.5 (to get rid of artifact)
                if ((peak_params[max_ampl_idx][0] < 15) &  \
                    (peak_params[max_ampl_idx][0] > 4) &  \
                    (peak_params[max_ampl_idx][1] >.2) &  \
                    (peak_params[max_ampl_idx][1] < 1.5)):
                    
                    # write oscillation parameters to dataframe
                    features_df['CF'][ii] = peak_params[max_ampl_idx][0]
                    features_df['Amp'][ii] = peak_params[max_ampl_idx][1]
                    features_df['BW'][ii] = peak_params[max_ampl_idx][2]
                                 
                # otherwise write empty
                else:   
                    features_df['CF'][ii] = np.nan
                    features_df['Amp'][ii] = np.nan
                    features_df['BW'][ii] = np.nan
                    
            # if no peaks are found, write empty
            elif len(peak_params) == 0:
                
                # write empty
                features_df['CF'][ii] = np.nan
                features_df['Amp'][ii] = np.nan
                features_df['BW'][ii] = np.nan
            
            # add backgr parameters to dataframe
            features_df['offset'][ii] = background_params[0]
            features_df['knee'][ii] = background_params[1]
            features_df['exp'][ii] = background_params[2]
            
            
            # get the long frequency range aperiodic parameters 
            # Initialize FOOOF model
            fm = FOOOF(peak_width_limits=bw_lims, background_mode='knee', max_n_peaks=max_n_peaks)
            
            # fit model with long range
            fm.fit(freq_mean, psd_mean, freq_range_long) 
            
            #offset, knee, slope of long range
            background_params_long = fm.background_params_
            
            # add long background parameters to dataframe
            features_df['offset_long'][ii] = background_params_long[0]
            features_df['knee_long'][ii] = background_params_long[1]
            features_df['exp_long'][ii] = background_params_long[2]
            
            print('this was ch', ii)

            
    return features_df
    

#%%

#%% 
    

def fooof_highest_peak_epoch_4_12_monkey(datastruct, fs, freq_range, bw_lims, max_n_peaks, freq_range_long, features_df):
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
        
    # initialize space in features_df     
    # periodic component
    features_df['CF'] = np.nan
    features_df['Amp'] = np.nan
    features_df['BW'] = np.nan
    
    # aperiodic component
    features_df['offset'] = np.nan
    features_df['knee'] = np.nan
    features_df['exp'] = np.nan
    
    # aperiodic component long frequency range
    features_df['offset_long'] = np.nan
    features_df['knee_long'] = np.nan
    features_df['exp_long'] = np.nan
    
    for ii in range(len(features_df)):
  
        # get data
        subj = features_df['subj'][ii]
        ch = features_df['ch'][ii]
        ep = features_df['ep'][ii]
        sig = datastruct[subj][ch][ep]
        
        # compute frequency spectrum
        freq_mean, psd_mean = spectral.compute_spectrum(sig, fs, method='welch', avg_type='mean', nperseg=fs*2)
        
        # if dead channel
        if sum(psd_mean) == 0: 
            
            # no oscillation was found
            features_df['CF'][ii] = np.nan
            features_df['Amp'][ii] = np.nan
            features_df['BW'][ii] = np.nan
            
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
                # 1) CF between 4 and 12 Hz
                # 2) Amp above .2
                if ((peak_params[max_ampl_idx][0] < 15) &  \
                    (peak_params[max_ampl_idx][0] > 4) &  \
                    (peak_params[max_ampl_idx][1] >.15)):
                    
                    # write oscillation parameters to dataframe
                    features_df['CF'][ii] = peak_params[max_ampl_idx][0]
                    features_df['Amp'][ii] = peak_params[max_ampl_idx][1]
                    features_df['BW'][ii] = peak_params[max_ampl_idx][2]
                                 
                # otherwise write empty
                else:   
                    features_df['CF'][ii] = np.nan
                    features_df['Amp'][ii] = np.nan
                    features_df['BW'][ii] = np.nan
                    
            # if no peaks are found, write empty
            elif len(peak_params) == 0:
                
                # write empty
                features_df['CF'][ii] = np.nan
                features_df['Amp'][ii] = np.nan
                features_df['BW'][ii] = np.nan
            
            # add backgr parameters to dataframe
            features_df['offset'][ii] = background_params[0]
            features_df['knee'][ii] = background_params[1]
            features_df['exp'][ii] = background_params[2]
            
            
            # get the long frequency range aperiodic parameters 
            # Initialize FOOOF model
            fm = FOOOF(peak_width_limits=bw_lims, background_mode='knee', max_n_peaks=max_n_peaks)
            
            # fit model with long range
            fm.fit(freq_mean, psd_mean, freq_range_long) 
            
            #offset, knee, slope of long range
            background_params_long = fm.background_params_
            
            # add long background parameters to dataframe
            features_df['offset_long'][ii] = background_params_long[0]
            features_df['knee_long'][ii] = background_params_long[1]
            features_df['exp_long'][ii] = background_params_long[2]
            
            print('this was ch', ii)

            
    return features_df
    
#%% 
    
def fooof_highest_peak_epoch_rat(datastruct, fs, freq_range, bw_lims, max_n_peaks, freq_range_long, features_df):
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
        
    # initialize space in features_df     
    # periodic component
    features_df['CF'] = np.nan
    features_df['Amp'] = np.nan
    features_df['BW'] = np.nan
    
    # aperiodic component
    features_df['offset'] = np.nan
    features_df['knee'] = np.nan
    features_df['exp'] = np.nan
    
    # aperiodic component long frequency range
    features_df['offset_long'] = np.nan
    features_df['knee_long'] = np.nan
    features_df['exp_long'] = np.nan
    
    for ii in range(len(features_df)):
  
        # get data
        subj = features_df['subj'][ii]
        day = features_df['day'][ii]
        ch = features_df['ch'][ii]
        ep = features_df['ep'][ii]
        
        if datastruct[subj][day][ch] is not None: 
            
            sig = datastruct[subj][day][ch][ep]
            
            # compute frequency spectrum
            freq_mean, psd_mean = spectral.compute_spectrum(sig, fs, method='welch', avg_type='mean', nperseg=fs*2)
            
            # if dead channel
            if sum(psd_mean) == 0: 
                
                # no oscillation was found
                features_df['CF'][ii] = np.nan
                features_df['Amp'][ii] = np.nan
                features_df['BW'][ii] = np.nan
                
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
                    # 1) CF between 4 and 12 Hz
                    # 2) Amp above .2
                    if ((peak_params[max_ampl_idx][0] < 15) &  \
                        (peak_params[max_ampl_idx][0] > 4) &  \
                        (peak_params[max_ampl_idx][1] >.15)):
                        
                        # write oscillation parameters to dataframe
                        features_df['CF'][ii] = peak_params[max_ampl_idx][0]
                        features_df['Amp'][ii] = peak_params[max_ampl_idx][1]
                        features_df['BW'][ii] = peak_params[max_ampl_idx][2]
                                     
                    # otherwise write empty
                    else:   
                        features_df['CF'][ii] = np.nan
                        features_df['Amp'][ii] = np.nan
                        features_df['BW'][ii] = np.nan
                        
                # if no peaks are found, write empty
                elif len(peak_params) == 0:
                    
                    # write empty
                    features_df['CF'][ii] = np.nan
                    features_df['Amp'][ii] = np.nan
                    features_df['BW'][ii] = np.nan
                
                # add backgr parameters to dataframe
                features_df['offset'][ii] = background_params[0]
                features_df['knee'][ii] = background_params[1]
                features_df['exp'][ii] = background_params[2]
                
                
                # get the long frequency range apSeriodic parameters 
                # Initialize FOOOF model
                fm = FOOOF(peak_width_limits=bw_lims, background_mode='knee', max_n_peaks=max_n_peaks)
                
                # fit model with long range
                fm.fit(freq_mean, psd_mean, freq_range_long) 
                
                #offset, knee, slope of long range
                background_params_long = fm.background_params_
                
                # add long background parameters to dataframe
                features_df['offset_long'][ii] = background_params_long[0]
                features_df['knee_long'][ii] = background_params_long[1]
                features_df['exp_long'][ii] = background_params_long[2]
                
                print('this was ch', ii)

            
    return features_df
    

#%%  Calculate PAC for monkey data - but phase providing band is variabel

def cal_pac_values_varphase_rat(datastruct, amplitude_providing_band, fs, features_df):
    """ iterates over all subjects and channels, calculates the PAC and results in dataframe
    Same function as cal_pac_values but:
        with a variable phase providing band
        and for the full timeframe instead of 1 second
        
    Inputs:
    -   Datastruct [subj * channels * data ] in numpy arrays
    -   Amplitude frequency band
    -   Sampling Frequency  
    -   Features df
    
    """  
 
    # create output columns
    features_df['pac_presence']  = np.int64
    features_df['pac_pvals'] = np.nan
    features_df['pac_rhos']  = np.nan

    for ii in range(len(features_df)):
            
        # for every channel that has peaks
        if ~np.isnan(features_df['CF'][ii]):
        
            # define phase providing band
            CF = features_df['CF'][ii]
            BW = features_df['BW'][ii]
            
            phase_providing_band= [(CF - (BW/2)),  (CF + (BW/2))]
            
            # get data
            subj = features_df['subj'][ii]
            day = features_df['day'][ii]
            ch = features_df['ch'][ii]
            ep = features_df['ep'][ii]

            data = datastruct[subj][day][ch][ep]
        
            #calculating phase of theta
            phase_data = pacf.butter_bandpass_filter(data, phase_providing_band[0], phase_providing_band[1], round(float(fs)));
            phase_data_hilbert = hilbert(phase_data);
            phase_data_angle = np.angle(phase_data_hilbert);
            
            #calculating amplitude envelope of high gamma
            amp_data = pacf.butter_bandpass_filter(data, amplitude_providing_band[0], amplitude_providing_band[1], round(float(fs)));
            amp_data_hilbert = hilbert(amp_data);
            amp_data_abs = abs(amp_data_hilbert);
            
            if ~np.isnan(phase_data_angle).any():
                PAC_values = pacf.circle_corr(phase_data_angle, amp_data_abs)
            
                if PAC_values[1] <= 0.05:
                    
                    features_df['pac_presence'][ii] = 1
                    
                elif PAC_values[1] > 0.05: 
        
                    features_df['pac_presence'][ii] = 0
                    
                features_df['pac_pvals'][ii] = PAC_values[1]
                features_df['pac_rhos'][ii] = PAC_values[0]
                
                print('this was ch', ii)
    
    return features_df

#%%
    
def resampled_pac_varphase_rat(datastruct, amplitude_providing_band, fs, num_resamples, features_df):
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
    # create output columns
    features_df['resamp_pac_presence']  = np.int64
    features_df['resamp_pac_pvals'] = np.nan
    features_df['resamp_pac_zvals']  = np.nan
    
    for ii in range(len(features_df)):
            
        # time it       
        start = time.time()
            
       
        # for every channel that has periodic peak
        if ~np.isnan(features_df['pac_rhos'][ii]):
        
            # define phase providing band
            CF = features_df['CF'][ii]
            BW = features_df['BW'][ii]
            
            phase_providing_band= [(CF - (BW/2)),  (CF + (BW/2))]
            
            # get data
            subj = features_df['subj'][ii]
            day = features_df['day'][ii]
            ch = features_df['ch'][ii]
            ep = features_df['ep'][ii]
    
            data = datastruct[subj][day][ch][ep]
    
            
            # create array of random numbers between 0 and total samples 
            # which is as long as the number of resamples
            roll_array = np.random.randint(0, len(data), size=num_resamples)
            
            resampled_pvals = np.full(num_resamples,np.nan)
            resampled_rhos = np.full(num_resamples,np.nan)
         
            # for every resample
            for jj in range(len(roll_array)):
    
                # roll the phase data for a random amount 
                phase_data_roll = np.roll(data, roll_array[jj])
                
                #calculating phase of theta
                phase_data = pacf.butter_bandpass_filter(phase_data_roll, phase_providing_band[0], phase_providing_band[1], round(float(fs)));
                phase_data_hilbert = hilbert(phase_data);
                phase_data_angle = np.angle(phase_data_hilbert);
                
                #calculating amplitude envelope of high gamma
                amp_data = pacf.butter_bandpass_filter(data, amplitude_providing_band[0], amplitude_providing_band[1], round(float(fs)));
                amp_data_hilbert = hilbert(amp_data);
                amp_data_abs = abs(amp_data_hilbert);
                
                if ~np.isnan(phase_data_angle).any():
                    
                    # calculate PAC
                    PAC_values = pacf.circle_corr(phase_data_angle, amp_data_abs)
                
                    resampled_pvals[jj] = PAC_values[1]
                    resampled_rhos[jj] = PAC_values[0]
            
            # after rolling data and calculating 1000 rho and pac's 
            # calculate the true values after resampling
            true_z = (features_df['pac_rhos'][ii] - np.mean(resampled_rhos)) / np.std(resampled_rhos)
            p_value = scipy.stats.norm.sf(abs(true_z))
         
            # write to dataframe
            features_df['resamp_pac_pvals'][ii] = p_value
            features_df['resamp_pac_zvals'][ii] = true_z
            
            # is it significant?
            if true_z >= 0  and  p_value <= 0.05:
    
                features_df['resamp_pac_presence'][ii] = 1         
                
            else: 
            
                features_df['resamp_pac_presence'][ii] = 0 
            
       
            print('this was ch', ii)
        
            end = time.time()
            print(end - start)   
        
    return features_df

#%%
    
def find_fooof_peaks(data_dict, freq_range, bw_lims, max_n_peaks, features_df, key):
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
        
    Outputs:
        Arrays of biggest peak characterics [CF, Ampl, BW]
        Array of background parameters [Exp, knee, offset]
    
    """

    
    # initialize space in features_df     
    # periodic component
    features_df[key]['CF'] = np.nan
    features_df[key]['Amp'] = np.nan
    features_df[key]['BW'] = np.nan
    
    # aperiodic component
    features_df[key]['offset'] = np.nan
    features_df[key]['knee'] = np.nan
    features_df[key]['exp'] = np.nan
    
    for ii in range(len(features_df[key])):
      
        
        signal, fs = get_signal(data_dict, features_df, key, ii)
        
    
        if ~np.isnan(signal).any(): 
            
            
            # compute frequency spectrum
            freq_mean, psd_mean = spectral.compute_spectrum(signal, fs, method='welch', avg_type='mean', nperseg=fs*2)
            
            # if dead channel
            if sum(psd_mean) == 0: 
                
                # no oscillation was found
                features_df[key]['CF'][ii] = np.nan
                features_df[key]['Amp'][ii] = np.nan
                features_df[key]['BW'][ii] = np.nan
                
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
                    # 1) CF between 4 and 12 Hz
                    # 2) Amp above .2
                    if ((peak_params[max_ampl_idx][0] < 24) &  \
                        (peak_params[max_ampl_idx][0] > 4) &  \
                        (peak_params[max_ampl_idx][1] >.15)):
                        
                        # write oscillation parameters to dataframe
                        features_df[key]['CF'][ii] = peak_params[max_ampl_idx][0]
                        features_df[key]['Amp'][ii] = peak_params[max_ampl_idx][1]
                        features_df[key]['BW'][ii] = peak_params[max_ampl_idx][2]
                                     
                    # otherwise write empty
                    else:   
                        features_df[key]['CF'][ii] = np.nan
                        features_df[key]['Amp'][ii] = np.nan
                        features_df[key]['BW'][ii] = np.nan
                        
                # if no peaks are found, write empty
                elif len(peak_params) == 0:
                    
                    # write empty
                    features_df[key]['CF'][ii] = np.nan
                    features_df[key]['Amp'][ii] = np.nan
                    features_df[key]['BW'][ii] = np.nan
                
                # add backgr parameters to dataframe
                features_df[key]['offset'][ii] = background_params[0]
                features_df[key]['knee'][ii] = background_params[1]
                features_df[key]['exp'][ii] = background_params[2]
                
                
                print('this was ch', ii)
    
    return features_df

#%%
    
def find_fooof_peaks_lowgamma(data_dict, freq_range, bw_lims, max_n_peaks, features_df, key):
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
        
    Outputs:
        Arrays of biggest peak characterics [CF, Ampl, BW]
        Array of background parameters [Exp, knee, offset]
    
    """

    
    # initialize space in features_df     
    # low gamma component
    features_df[key]['LG_CF'] = np.nan
    features_df[key]['LG_Amp'] = np.nan
    features_df[key]['LG_BW'] = np.nan
    
    
    for ii in range(len(features_df[key])):
        
        signal, fs = get_signal(data_dict, features_df, key, ii)
        
    
        if ~np.isnan(signal).any(): 
            
            
            # compute frequency spectrum
            freq_mean, psd_mean = spectral.compute_spectrum(signal, fs, method='welch', avg_type='mean', nperseg=fs*2)
            
            # if dead channel
            if sum(psd_mean) == 0: 
                
                features_df[key]['LG_CF'][ii] = np.nan
                features_df[key]['LG_Amp'][ii] = np.nan
                features_df[key]['LG_BW'][ii] = np.nan
                
            else:
                
                # Initialize FOOOF model
                fm = FOOOF(peak_width_limits=bw_lims, background_mode='knee', max_n_peaks=max_n_peaks)
                
                # fit model
                fm.fit(freq_mean, psd_mean, freq_range)    
                
                # Central frequency, Amplitude, Bandwidth
                peak_params = fm.peak_params_                
                
                # for monkey dont take the 48-52 hz due to line noise
                if key == 'monkey': 
                    
                    peak_params = np.squeeze([
                            peak_params[ii] for ii in range(len(peak_params)) 
                            if (((peak_params[ii][0] > 30) & (peak_params[ii][0] < 48)) or 
                            ((peak_params[ii][0] > 52) & (peak_params[ii][0] < 58)))    
                            ])
                    
                    
                else: 
                    
                    peak_params = np.squeeze([
                            peak_params[ii] for ii in range(len(peak_params)) 
                            if ((peak_params[ii][0] > 30) & (peak_params[ii][0] < 58))
                            ])
                    
     
                # if peaks are found
                if len(peak_params) > 0: 
                    
                    if len(np.shape(peak_params)) > 1:
                        
                        # find which peak has the biggest amplitude
                        max_ampl_idx = np.argmax(peak_params[:,1])
                        
                        # find this peak hase the following characteristics:
                        # Amp above .15
                        if peak_params[max_ampl_idx][1] >.15:
                            
                            # write oscillation parameters to dataframe
                            features_df[key]['LG_CF'][ii] = peak_params[max_ampl_idx][0]
                            features_df[key]['LG_Amp'][ii] = peak_params[max_ampl_idx][1]
                            features_df[key]['LG_BW'][ii] = peak_params[max_ampl_idx][2]
                                         
                        # otherwise write empty
                        else:   
                            features_df[key]['LG_CF'][ii] = np.nan
                            features_df[key]['LG_Amp'][ii] = np.nan
                            features_df[key]['LG_BW'][ii] = np.nan
                        
                    elif len(np.shape(peak_params)) == 1:
                        
                        # find this peak hase the following characteristics:
                        # Amp above .15
                        if peak_params[1] >.15:
                            
                            # write oscillation parameters to dataframe
                            features_df[key]['LG_CF'][ii] = peak_params[0]
                            features_df[key]['LG_Amp'][ii] = peak_params[1]
                            features_df[key]['LG_BW'][ii] = peak_params[2]
                                         
                        # otherwise write empty
                        else:   
                            features_df[key]['LG_CF'][ii] = np.nan
                            features_df[key]['LG_Amp'][ii] = np.nan
                            features_df[key]['LG_BW'][ii] = np.nan
                            
                # if no peaks are found, write empty
                elif len(peak_params) == 0:
                    
                    # write empty
                    features_df[key]['LG_CF'][ii] = np.nan
                    features_df[key]['LG_Amp'][ii] = np.nan
                    features_df[key]['LG_BW'][ii] = np.nan

            print('this was ch', ii)
    
    return features_df
#%% Calculate PAC for all three datasets
    
def calculate_pac(data_dict, amplitude_providing_band, features_df, key):
    """ iterates over all subjects and channels, calculates the PAC and results in dataframe
    Same function as cal_pac_values but:
        with a variable phase providing band
        and for the full timeframe instead of 1 second
        
    Inputs:
    -   Datastruct [subj * channels * data ] in numpy arrays
    -   Amplitude frequency band
    -   Sampling Frequency  
    -   Features df
    
    """  
 
    # create output columns
    features_df[key]['pac_presence']  = np.int64
    features_df[key]['pac_pvals'] = np.nan
    features_df[key]['pac_rhos']  = np.nan

    for ii in range(len(features_df[key])):
            
        # for every channel that has peaks
        if ~np.isnan(features_df[key]['CF'][ii]):
        
            # define phase providing band
            CF = features_df[key]['CF'][ii]
            BW = features_df[key]['BW'][ii]
            
            phase_providing_band= [(CF - (BW/2)),  (CF + (BW/2))]
            
            # get singal and fs
            signal, fs = get_signal(data_dict, features_df, key, ii)
        
            #calculating phase of theta
            phase_data = pacf.butter_bandpass_filter(signal, phase_providing_band[0], phase_providing_band[1], fs);
            phase_data_hilbert = hilbert(phase_data);
            phase_data_angle = np.angle(phase_data_hilbert);
            
            #calculating amplitude envelope of high gamma
            amp_data = pacf.butter_bandpass_filter(signal, amplitude_providing_band[0], amplitude_providing_band[1], fs);
            amp_data_hilbert = hilbert(amp_data);
            amp_data_abs = abs(amp_data_hilbert);
            
            if ~np.isnan(phase_data_angle).any():
                
                PAC_values = pacf.circle_corr(phase_data_angle, amp_data_abs)
            
                if PAC_values[1] <= 0.05:
                    
                    features_df[key]['pac_presence'][ii] = 1
                    
                elif PAC_values[1] > 0.05: 
        
                    features_df[key]['pac_presence'][ii] = 0
                    
                features_df[key]['pac_pvals'][ii] = PAC_values[1]
                features_df[key]['pac_rhos'][ii] = PAC_values[0]
                
                print('this was ch', ii)
    
    return features_df

#%% Calculate PAC low gamma for all three datasets
    
def calculate_pac_lowgamma(data_dict, features_df, key):

 
    # create output columns
    features_df[key]['LG_pac_presence']  = np.int64
    features_df[key]['LG_pac_pvals'] = np.nan
    features_df[key]['LG_pac_rhos']  = np.nan

    for ii in range(len(features_df[key])):
            
        # for every channel that has peaks
        if (~np.isnan(features_df[key]['CF'][ii]) &
            ~np.isnan(features_df[key]['LG_CF'][ii])):
        
            # define phase providing band
            CF = features_df[key]['CF'][ii]
            BW = features_df[key]['BW'][ii]
            
            phase_providing_band= [(CF - (BW/2)),  (CF + (BW/2))]
            
            # define amplitude providing band
            CF_lowgamma = features_df[key]['LG_CF'][ii]
            BW_lowgamma = features_df[key]['LG_BW'][ii]
            
            amplitude_providing_band= [(CF_lowgamma - (BW_lowgamma/2)),  (CF_lowgamma + (BW_lowgamma/2))]
            
            # get singal and fs
            signal, fs = get_signal(data_dict, features_df, key, ii)
        
            #calculating phase of theta
            phase_data = pacf.butter_bandpass_filter(signal, phase_providing_band[0], phase_providing_band[1], fs);
            phase_data_hilbert = hilbert(phase_data);
            phase_data_angle = np.angle(phase_data_hilbert);
            
            #calculating amplitude envelope of high gamma
            amp_data = pacf.butter_bandpass_filter(signal, amplitude_providing_band[0], amplitude_providing_band[1], fs);
            amp_data_hilbert = hilbert(amp_data);
            amp_data_abs = abs(amp_data_hilbert);
            
            if ~np.isnan(phase_data_angle).any():
                
                PAC_values = pacf.circle_corr(phase_data_angle, amp_data_abs)
            
                if PAC_values[1] <= 0.05:
                    
                    features_df[key]['LG_pac_presence'][ii] = 1
                    
                elif PAC_values[1] > 0.05: 
        
                    features_df[key]['LG_pac_presence'][ii] = 0
                    
                features_df[key]['LG_pac_pvals'][ii] = PAC_values[1]
                features_df[key]['LG_pac_rhos'][ii] = PAC_values[0]
                
                print('this was ch', ii)
    
    return features_df    

