# version 2.0.0
from neurodsp import sim, spectral
from neurodsp.utils import create_times

# version 0.1.3
from fooof import FOOOF

# version 0.1.2
from bycycle.filt import lowpass_filter
from bycycle.features import compute_features

import random
from scipy.stats import truncnorm
import numpy as np
import scipy as sp
from scipy.signal import butter, filtfilt
import time
import os
from scipy.signal import hilbert
import pandas as pd

pd.options.mode.chained_assignment = None

#%% Functions

def butter_bandpass(lowcut, highcut, fs, order=4):
    """lowcut is the lower bound of the frequency that we want to isolate
    hicut is the upper bound of the frequency that we want to isolate
    fs is the sampling rate of our data"""
    nyq = 0.5 * fs #nyquist frequency - see http://www.dspguide.com/ if you want more info
    low = float(lowcut) / nyq
    high = float(highcut) / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(mydata, lowcut, highcut, fs, order=4):
    """ builds on butter_bandpass, uses filtfilt and returns data"""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, mydata)
    return y

def circle_corr(ang,line):
    """give amplitude and phase data as input, returns correlation stats"""
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

#%% CORRECT VERSION
os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab')

fs = 1000
n_seconds = 32 # if higher, original PSD will be stronger so maybe less brown noise
freq_list = [10] #[5, 10, 20]
lf_waveform = ['sine', 'asine']  
lf_rdsym = [.1, .2, .3, .4, .6, .7, .8, .9]
plt_time =  [0,3000]
hf_range = [80,250]
exp = [-2, -1.5]
times = create_times(n_seconds, fs)

# parameters for HG firing distributions 
mean = 0.5 
sd_list = np.linspace(5,29,25) / 100
uniform = np.array([1,1,1,1,1])
sd_list = np.append(sd_list, uniform)
low_bound = 0
upp_bound = 1

# calculate length signal
signal_length = n_seconds * fs

# FOOOF parameters
freq_range = [4, 250] # for peak detection
bw_lims = [1, 2]
max_n_peaks = 26
min_peak_amplitude = .7

# ByCycle parameters
f_lowpass = 45
n_seconds_kernal = 2
burst_kwargs = {'amplitude_fraction_threshold': .3,
                'amplitude_consistency_threshold': .4,
                'period_consistency_threshold': .5,
                'monotonicity_threshold': .8,
                'N_cycles_min': 10}

# number of iterations
n_it = 1500

# set up dataframe
simulation_features = pd.DataFrame(index=range(0,n_it))
           
# put values into dataframe
simulation_features['it'] = np.int64
simulation_features['freq'] = np.int64
simulation_features['asine_rdsym'] = np.nan
simulation_features['noise_exp'] = np.nan
simulation_features['firing_std'] = np.nan
simulation_features['offset_before'] = np.nan
simulation_features['exp_before'] = np.nan
simulation_features['offset'] = np.nan
simulation_features['exp'] = np.nan
simulation_features['volt_amp'] = np.nan
simulation_features['rdsym'] = np.nan
simulation_features['ptsym'] = np.nan
simulation_features['pac_rhos'] = np.nan
simulation_features['pac_presence'] = np.int64
  


for aa in range(0,n_it):   
    

    # time it       
    start = time.time()

    
    # simulate oscillation
    # get random oscillation between frequency ranges
    # osc_freq_rand = random.randint(osc_freq[0], osc_freq[1])
    osc_freq_rand = random.choice(freq_list)
    
    # if sinusiod
    asine_or_sine = random.choice(lf_waveform)
    
    if asine_or_sine == 'sine':
   
        signal = sim.sim_oscillation(n_seconds, fs, osc_freq_rand, cycle='sine')
        
        simulation_features['asine_rdsym'][aa] = .5
        
    elif asine_or_sine == 'asine':
        
        asine_rdsym = random.choice(lf_rdsym)
        signal = sim.sim_oscillation(n_seconds, fs, osc_freq_rand, cycle='asine' ,rdsym=asine_rdsym)
        
        simulation_features['asine_rdsym'][aa] = asine_rdsym
    
    # roll signal to have different phase every simulation
    signal = np.roll(signal, random.randint(0,signal_length))
    
    # get random exponential 
    exp_rand = random.uniform(exp[0], exp[1])
    
    # simulate 50x brown noise on top to get reliable slope
    n_brown_noise = 50
    for br in range(n_brown_noise):
        br_noise = sim.sim_powerlaw(n_seconds, fs, exp_rand)
        signal = signal + br_noise
        
    try:
        
        
        # run FOOOF
        # compute frequency spectrum
        freq_mean, psd_mean = spectral.compute_spectrum(signal, fs, method='welch', avg_type='mean', nperseg=fs*2)
        
        # Initialize FOOOF model
        fm = FOOOF(peak_width_limits=bw_lims, background_mode='fixed', max_n_peaks=max_n_peaks,
                   min_peak_amplitude=min_peak_amplitude)
    
        # fit model
        fm.fit(freq_mean, psd_mean, freq_range)
        
        simulation_features['offset_before'][aa] = fm.background_params_[0]
        simulation_features['exp_before'][aa] = fm.background_params_[1]
        
        
        # get distribution
        sd = random.choice(sd_list)
        
        # creat distribution function
        hf_distribution = truncnorm((low_bound - mean) / sd, (upp_bound - mean) / sd, loc=mean, scale=sd)
        
        # calculate length cycle
        samples_cycle = 1/osc_freq_rand * fs
        
        # create array with sample in which each cycle starts
        cycle_array = np.linspace(0,signal_length, osc_freq_rand * n_seconds, endpoint=False)
        
        # calculate number of bursts per cycle - 1000 bursts per second
        total_bursts_sec = 1000 * n_seconds
        bursts_per_cycle = int(total_bursts_sec / osc_freq_rand)
        
        # cc = for each cycle 
        for cc in range(1,len(cycle_array)-1):
                        
            # 100 samples of the distribution for each cycle in signal
            hf_distribution_array = hf_distribution.rvs(bursts_per_cycle)
            
            # translate the distributions to each sample were activity is
            hf_activity_array = [(cycle_array[cc] + (hf_distribution_array[ii] * samples_cycle)) 
                                for ii in range(len(hf_distribution_array))]   
            
            # kk = for each simulated high freq activity
            for kk in range(len(hf_activity_array)):
                
                # get random high frequency and length of a cycle of that frequency
                hf_freq = random.randint(hf_range[0],hf_range[1])
                n_seconds_cycle = (1/hf_freq * fs)/fs
                
                # create cycle of high frequency with 0.05 'power'
                hf_cycle = np.sin(2*np.pi*1/n_seconds_cycle * (np.arange(fs*n_seconds_cycle)/fs))*.01
                
                # add cycles to zeros signal - in the center
                signal[int(round(hf_activity_array[kk]) -  np.ceil(len(hf_cycle) / 2)) : 
                    int(round(hf_activity_array[kk]) + np.floor(len(hf_cycle) / 2))] = \
                    signal[int(round(hf_activity_array[kk]) -  np.ceil(len(hf_cycle) / 2)) : \
                    int(round(hf_activity_array[kk]) + np.floor(len(hf_cycle) / 2))] + hf_cycle
          
        # cut off first second and last second of signal
        signal = signal[fs*1:-fs*1]
        
        # plot_time_series(times[plt_time[0]:plt_time[1]], signal[plt_time[0]:plt_time[1]])
        
        # run FOOOF
        # compute frequency spectrum
        freq_mean, psd_mean = spectral.compute_spectrum(signal, fs, method='welch', avg_type='mean', nperseg=fs*2)
        
        # Initialize FOOOF model
        fm = FOOOF(peak_width_limits=bw_lims, background_mode='fixed', max_n_peaks=max_n_peaks,
                   min_peak_amplitude=min_peak_amplitude)
    
        # fit model
        fm.fit(freq_mean, psd_mean, freq_range)    
    
        
        # run ByCycle
        # low pass filter
        lp_signal = lowpass_filter(signal, fs, f_lowpass, N_seconds=n_seconds_kernal, remove_edge_artifacts=False)
        
        # bycycle dataframe            
        bycycle_df = compute_features(lp_signal, fs, [osc_freq_rand - 2, osc_freq_rand + 2],  burst_detection_kwargs=burst_kwargs)
        
        
        # calculate PAC
        #calculating phase of theta
        phase_data = butter_bandpass_filter(signal, osc_freq_rand - 1, osc_freq_rand + 1, fs);
        phase_data_hilbert = hilbert(phase_data);
        phase_data_angle = np.angle(phase_data_hilbert);
        
        #calculating amplitude envelope of high gamma
        amp_data = butter_bandpass_filter(signal, 80, 250, fs);
        amp_data_hilbert = hilbert(amp_data);
        amp_data_abs = abs(amp_data_hilbert);
        
        PAC_values = circle_corr(phase_data_angle, amp_data_abs)
        
        # put values into dataframe
        simulation_features['it'][aa] = aa
        simulation_features['freq'][aa] = osc_freq_rand
        simulation_features['noise_exp'][aa] = exp_rand
        simulation_features['firing_std'][aa] = sd
        simulation_features['offset'][aa] = fm.background_params_[0]
        simulation_features['exp'][aa] = fm.background_params_[1]
        simulation_features['volt_amp'][aa] = bycycle_df['volt_peak'].median()
        simulation_features['rdsym'][aa] = bycycle_df['time_rdsym'].median()
        simulation_features['ptsym'][aa] = bycycle_df['time_ptsym'].median()
        simulation_features['pac_rhos'][aa] = PAC_values[0]
        
        if PAC_values[1] <= 0.05:
            
            simulation_features['pac_presence'][aa] = 1
            
        elif PAC_values[1] > 0.05: 
        
            simulation_features['pac_presence'][aa] = 0
    except:
        
        pass
    
    end = time.time()
    print(end - start)  

    if aa % 100==0:
        
        # 2_0_0 = exp [-2, -1.5], asine/sine, uniform/distribution STDs
        simulation_features.to_csv('simulation_features_2_0_0.csv', sep=',', index=False)
        print('We have ' + str(aa) + ' iterations')


