#%% Import standard libraries
import os
import numpy as np


#%% Import Voytek Lab libraries



#%% Change directory to import necessary modules

os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab\Code\distinguish_pac')

#%% import self-defined modules

import module_load_data as load_data
import module_pac_functions as pacf
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
            sig = datastruct[subj][ch][(ep*fs*epoch_len):((ep*fs*epoch_len)+fs*epoch_len)]
            
            # compute frequency spectrum
            freq_mean, psd_mean = spectral.compute_spectrum(sig, fs, method='welch', avg_type='mean', nperseg=fs*2)
         
            # Set the frequency range upon which to fit FOOOF
            freq_range = [4, 58]
            bw_lims = [2, 6]
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

from scipy.signal import hilbert


amplitude_providing_band = [62, 100]; #80-125 Hz band

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
                
                data =  datastruct[subj][ch][(ep*fs*epoch_len):((ep*fs*epoch_len)+fs*epoch_len)]
                
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
            
            # if no peak, assign nan's 
            if len(psd_peaks[subj][ch][ep]) == 0:
            
                resampled_pvalues_ep.append(np.full(num_resamples,np.nan))
                resampled_rhovalues_ep.append(np.full(num_resamples,np.nan))      
            
            
            # for every channel that has peaks
            elif len(psd_peaks[subj][ch][ep]) > 0:
            
                # define phase providing band
                CF = psd_peaks[subj][ch][ep][0]
                BW = psd_peaks[subj][ch][ep][2]
                
                phase_providing_band = [(CF - (BW/2)),  (CF + (BW/2))]
            
                # get data of that channel
                data = datastruct[subj][ch][(ep*fs*epoch_len):((ep*fs*epoch_len)+fs*epoch_len)]
                
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



#%% Find the true PAC values 
import scipy.io

pac_true_zvals = []
pac_true_pvals = []
pac_true_presence = []

# for every subj
for subj in range(len(resampled_rhovalues)):
    
    pac_true_zvals_ch = []
    pac_true_pvals_ch = []
    pac_true_presence_ch = []
    
    # for every channel
    for ch in range(len(resampled_rhovalues[subj])):
        
        pac_true_zvals_ep = np.full(5,np.nan)
        pac_true_pvals_ep = np.full(5,np.nan)
        pac_true_presence_ep = np.full(5,np.nan)
         
        for ep in range(num_epochs):
        
            true_z = (pac_rhos[subj][ch][ep] - np.mean(resampled_rhovalues[subj][ch][ep]))  \
                    / np.std(resampled_rhovalues[subj][ch][ep])
            p_value = scipy.stats.norm.sf(abs(true_z))
                
            pac_true_pvals_ep[ep] = p_value
            pac_true_zvals_ep[ep] = true_z
            
            if true_z >= 0  and  p_value <= 0.05:
    
                pac_true_presence_ep[ep] = 1         
                
            else: 
             
                pac_true_presence_ep[ep] = 0 
                
         
        pac_true_zvals_ch.append(pac_true_zvals_ep)
        pac_true_pvals_ch.append(pac_true_pvals_ep)
        pac_true_presence_ch.append(pac_true_presence_ep) 


    pac_true_pvals.append(pac_true_pvals_ch) 
    pac_true_zvals.append(pac_true_zvals_ch)
    pac_true_presence.append(pac_true_presence_ch)


#%% Now, we only select those channels that have a CF of < 15 Hz, 
### and with an amplitude between .2 & 1.5
### call these subj_idx and ch_idx and ep_idx
   
subj_idx = []
ch_idx = []
ep_idx = []    

for subj in range(len(resampled_rhovalues)):
    for ch in range(len(resampled_rhovalues[subj])):
        for ep in range(num_epochs):
           
            if len(psd_peaks[subj][ch][ep]) > 0: 
                
                if ((pac_true_presence[subj][ch][ep] == 1) & \
                    (psd_peaks[subj][ch][ep][0] < 15) & \
                    (psd_peaks[subj][ch][ep][1] > .2) & \
                    (psd_peaks[subj][ch][ep][1] < 1.5)):
                                      
                    subj_idx.append(subj)
                    ch_idx.append(ch)
                    ep_idx.append(ep)


#%% Loop over all channels with a CF of < 15 Hz, and with an AMP between .2 & 1.5
### And extract and save the cycle-by-cycle features

# create empty output
rdsym = []
ptsym = []
bursts = []
period = []
volt_amp = []

burst_kwargs = {'amplitude_fraction_threshold': 0.25,
                'amplitude_consistency_threshold': .4,
                'period_consistency_threshold': .45,
                'monotonicity_threshold': .6,
                'N_cycles_min': 3}

# for every channel with pac
for ii in range(len(subj_idx)):
    
    # get subj & ch
    subj = subj_idx[ii]
    ch = ch_idx[ii]
    ep = ep_idx[ii]
      
    # get phase providing band
    lower_phase = psd_peaks[subj][ch][ep][0] - (psd_peaks[subj][ch][ep][2] / 2)
    upper_phase = psd_peaks[subj][ch][ep][0] + (psd_peaks[subj][ch][ep][2] / 2)
    
    fs = 1000
    f_range = [lower_phase, upper_phase]
    f_lowpass = 55
    N_seconds = len(datastruct[subj][ch]) / fs / num_epochs - 2
    
    signal = lowpass_filter(datastruct[subj][ch][(ep*fs*epoch_len):((ep*fs*epoch_len)+fs*epoch_len)], fs, f_lowpass, N_seconds=N_seconds, remove_edge_artifacts=False)
    
    df = compute_features(signal, fs, f_range,  burst_detection_kwargs=burst_kwargs)
    
    is_burst = df['is_burst'].tolist()
    time_rdsym = df['time_rdsym'].to_numpy()
    time_ptsym = df['time_ptsym'].to_numpy()
    period_ch = df['period'].to_numpy()
    volt_amp_ch = df['volt_amp'].to_numpy()
    
    bursts.append(is_burst)
    rdsym.append(time_rdsym)
    ptsym.append(time_ptsym)
    period.append(period_ch)
    volt_amp.append(volt_amp_ch)

                
#%% 
    
clean_data = []
clean_pac_rhos = []
clean_resamp_zvals = []
clean_resamp_pvals = []
clean_psd_params = []
clean_backgr_params = []


for ii in range(len(subj_idx)):
    
    subj = subj_idx[ii]
    ch = ch_idx[ii]
    ep = ep_idx[ii]
         
    clean_data.append( datastruct[subj][ch][(ep*fs*epoch_len):((ep*fs*epoch_len)+fs*epoch_len)])
    clean_pac_rhos.append(pac_rhos[subj][ch][ep])
    clean_resamp_zvals.append(pac_true_zvals[subj][ch][ep])
    clean_resamp_pvals.append(pac_true_pvals[subj][ch][ep]) 
    clean_psd_params.append(psd_peaks[subj][ch][ep])
    clean_backgr_params.append(backgr_params[subj][ch][ep])
    
    
#%% Write to clean_db_20s
    
clean_db = {}

# metadata
clean_db['subj_name'] = subjects # list of initials of subjects (subjects)
clean_db['subj'] = subj_idx # array of the subjects to which data belong (pac_idx[0])
clean_db['ch'] = ch_idx # array of the channels to which data belong (pac_idx[1])
clean_db['ep'] = ep_idx
clean_db['locs'] = elec_locs
clean_db['dat_name'] = 'fixation_pwrlaw'
clean_db['fs'] = 1000

# data
clean_db['data'] = clean_data# list of arrays of channels with PAC

# analysis 
clean_db['pac_rhos'] = clean_pac_rhos # now array, put in array with only sig PAC chs
clean_db['resamp_zvals'] = clean_resamp_zvals # is now matrix, put in array with only sig PAC chs
clean_db['resamp_pvals'] = clean_resamp_pvals # is now matrix, put in array with only sig PAC chs

clean_db['psd_params'] = clean_psd_params # list of arrays,  put in array with only sig PAC chs
clean_db['backgr_params'] = clean_backgr_params # list of arrays,  put in array with only sig PAC chs
clean_db['rd_sym'] = rdsym # list of arrays,  put in array with only sig PAC chs
clean_db['pt_sym'] = ptsym # list of arrays,  put in array with only sig PAC chs
clean_db['bursts'] = bursts # list of arrays,  put in array with only sig PAC chs
clean_db['period'] = period # list of arrays,  put in array with only sig PAC chs
clean_db['volt_amp'] = volt_amp # list of arrays,  put in array with only sig PAC chs

# Save with pickle
import pickle

save_data = open("clean_db_20s_2.pkl","wb")
pickle.dump(clean_db,save_data)
save_data.close()





