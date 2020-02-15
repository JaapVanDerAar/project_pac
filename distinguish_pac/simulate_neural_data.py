from neurodsp import sim, spectral
from neurodsp.utils import create_times
from neurodsp.plts.time_series import plot_time_series
from fooof import FOOOF
from bycycle.filt import lowpass_filter
from bycycle.features import compute_features
import random
from scipy.stats import truncnorm
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import pandas as pd


os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab\Code\distinguish_pac')
import module_pac_functions as pacf

pd.options.mode.chained_assignment = None


#%% CORRECT VERSION

fs = 1000
n_seconds = 32 # if higher, original PSD will be stronger so maybe less brown noise
freq_list = [5, 10, 20]
plt_time =  [0,3000]
hf_range = [80,250]
exp = [-2, -1]
times = create_times(n_seconds, fs)

# parameters for HG firing distributions 
mean = 0.5 
sd_list = [0.01, 0.04, 0.16, 0.64] 
low_bound = 0
upp_bound = 1

# calculate length signal
signal_length = n_seconds * fs

# FOOOF parameters
freq_range = [4, 250] # for peak detection
bw_lims = [1, 2]
max_n_peaks = 20

# ByCycle parameters
f_lowpass = 45
n_seconds_kernal = 2
burst_kwargs = {'amplitude_fraction_threshold': .3,
                'amplitude_consistency_threshold': .4,
                'period_consistency_threshold': .5,
                'monotonicity_threshold': .8,
                'N_cycles_min': 10}

# number of iterations
n_it = 1000

# set up dataframe
simulation_features = pd.DataFrame(index=range(0,n_it))
           
# put values into dataframe
simulation_features['it'] = np.int64
simulation_features['freq'] = np.int64
simulation_features['noise_exp'] = np.nan
simulation_features['firing_std'] = np.nan
simulation_features['offset'] = np.nan
simulation_features['exp'] = np.nan
simulation_features['volt_amp'] = np.nan
simulation_features['rdsym'] = np.nan
simulation_features['ptsym'] = np.nan
simulation_features['pac_rhos'] = np.nan
simulation_features['pac_presence'] = np.int64
    

for aa in range(872,n_it):   
    
    # time it       
    start = time.time()
    
    # simulate oscillation
    # get random oscillation between frequency ranges
    # osc_freq_rand = random.randint(osc_freq[0], osc_freq[1])
    osc_freq_rand = random.choice(freq_list)
    osc_sine = sim.sim_oscillation(n_seconds, fs, osc_freq_rand, cycle='sine')
    
    signal = osc_sine
    
    # roll signal to have different phase every simulation
    signal = np.roll(signal, random.randint(0,signal_length))
    
    # get random exponential 
    exp_rand = random.uniform(exp[0], exp[1])
    
    # simulate 50x brown noise on top to get reliable slope
    n_brown_noise = 50
    for br in range(n_brown_noise):
        br_noise = sim.sim_powerlaw(n_seconds, fs, exp_rand)
        signal = signal + br_noise
    
    
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
            hf_cycle = np.sin(2*np.pi*1/n_seconds_cycle * (np.arange(fs*n_seconds_cycle)/fs))*.05
            
            # add cycles to zeros signal - in the center
            signal[int(round(hf_activity_array[kk]) -  np.ceil(len(hf_cycle) / 2)) : 
                int(round(hf_activity_array[kk]) + np.floor(len(hf_cycle) / 2))] = \
                signal[int(round(hf_activity_array[kk]) -  np.ceil(len(hf_cycle) / 2)) : \
                int(round(hf_activity_array[kk]) + np.floor(len(hf_cycle) / 2))] + hf_cycle
      
    # cut off first second and last second of signal
    signal = signal[fs*1:-fs*1]
    
    
    # run FOOOF
    # compute frequency spectrum
    freq_mean, psd_mean = spectral.compute_spectrum(signal, fs, method='welch', avg_type='mean', nperseg=fs*2)
    
    # Initialize FOOOF model
    fm = FOOOF(peak_width_limits=bw_lims, background_mode='fixed', max_n_peaks=max_n_peaks)
    
    
    try:
        # fit model
        fm.fit(freq_mean, psd_mean, freq_range)    
        
        
        # run ByCycle
        # low pass filter
        lp_signal = lowpass_filter(signal, fs, f_lowpass, N_seconds=n_seconds_kernal, remove_edge_artifacts=False)
        
        # bycycle dataframe            
        bycycle_df = compute_features(lp_signal, fs, [osc_freq_rand - 1, osc_freq_rand +1],  burst_detection_kwargs=burst_kwargs)
        
        
        # calculate PAC
        #calculating phase of theta
        phase_data = pacf.butter_bandpass_filter(signal, osc_freq_rand - 1, osc_freq_rand +1, fs);
        phase_data_hilbert = hilbert(phase_data);
        phase_data_angle = np.angle(phase_data_hilbert);
        
        #calculating amplitude envelope of high gamma
        amp_data = pacf.butter_bandpass_filter(signal, 80, 250, fs);
        amp_data_hilbert = hilbert(amp_data);
        amp_data_abs = abs(amp_data_hilbert);
        
        PAC_values = pacf.circle_corr(phase_data_angle, amp_data_abs)
        
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

os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab')
simulation_features.to_csv('simulation_features.csv', sep=',', index=False)


# sanity checks:
# - attenuation
# - what does Low pass filtering with bycycle? why is there shape if we add only HF's 
# - Fooof min_amp_peaks









#%% Basic parameters

fs = 1000
n_seconds = 100
osc_freq = 7
plt_time =  [0,1000]
times = create_times(n_seconds, fs)

#%% create and plot LF oscillation
osc_sine = sim.sim_oscillation(n_seconds, fs, osc_freq, cycle='sine')
plot_time_series(times[plt_time[0]:plt_time[1]], osc_sine[plt_time[0]:plt_time[1]])


#%% create and plot bursts

enter_burst = .4
leave_burst = .4

osc_burst = sim.sim_bursty_oscillation(n_seconds, fs, osc_freq,
                                 enter_burst=enter_burst,
                                 leave_burst=leave_burst)
plot_time_series(times[plt_time[0]:plt_time[1]], osc_burst[plt_time[0]:plt_time[1]])


#%% create brown 1/f noise

exponent = -1

br_noise = sim.sim_powerlaw(n_seconds, fs, exponent)

plot_time_series(times[plt_time[0]:plt_time[1]], br_noise[plt_time[0]:plt_time[1]])


#%% create brown 1/f noise with high pass 1 Hz

f_hipass_brown = 1

br_noise_filt = sim.sim_powerlaw(n_seconds, fs, exponent, f_range=(f_hipass_brown, None))

plot_time_series(times[plt_time[0]:plt_time[1]], br_noise_filt[plt_time[0]:plt_time[1]])


#%% Synaptic current signal

syn_current = sim.sim_synaptic_current(n_seconds, fs, n_neurons=1000, 
                                 firing_rate=.2, tau_r=0, tau_d=0.01, t_ker=None)

plot_time_series(times[plt_time[0]:plt_time[1]], syn_current[plt_time[0]:plt_time[1]])

#%% Poisson activity

poisson_pop = sim.sim_poisson_pop(n_seconds, fs, n_neurons=1000, firing_rate=2)

plot_time_series(times[plt_time[0]:plt_time[1]], poisson_pop[plt_time[0]:plt_time[1]])


#%% FOOOF

freq_range = [4, 150] # for peak detection
bw_lims = [2, 6]
max_n_peaks = 5
            
signal = osc_sine + br_noise

plot_time_series(times[plt_time[0]:plt_time[1]], signal[plt_time[0]:plt_time[1]])


# compute frequency spectrum
freq_mean, psd_mean = spectral.compute_spectrum(signal, fs, method='welch', avg_type='mean', nperseg=fs*2)

# Initialize FOOOF model
fm = FOOOF(peak_width_limits=bw_lims, background_mode='knee', max_n_peaks=max_n_peaks)

# fit model
fm.fit(freq_mean, psd_mean, freq_range)    

fm.report()


#%%

f_lowpass = 20
N_seconds = 2
fs = 1000
burst_kwargs = {'amplitude_fraction_threshold': .3,
                'amplitude_consistency_threshold': .4,
                'period_consistency_threshold': .5,
                'monotonicity_threshold': .8,
                'N_cycles_min': 3}

            

lp_signal = lowpass_filter(signal, fs, f_lowpass, N_seconds=N_seconds, remove_edge_artifacts=False)

bycycle_df = compute_features(lp_signal, fs, [osc_freq-2, osc_freq+2],  burst_detection_kwargs=burst_kwargs)

#%% 

# plt.figure(figsize=(40,40)) 
plt.hist(bycycle_df['time_ptsym'])

print(bycycle_df['volt_peak'].median())
print(bycycle_df['time_rdsym'].median())
print(bycycle_df['time_ptsym'].median())

plot_burst_detect_params(signal, fs, bycycle_df,
                 burst_kwargs, tlims=(0, 5), figsize=(16, 3))


#%%


fs = 1000
n_seconds = 100
osc_freq = [4,12]
plt_time =  [0,3000]

exp = [-3, -1]


times = create_times(n_seconds, fs)



#%% create and plot LF oscillation

# simulate oscillation
# get random oscillation between frequency ranges
osc_freq_rand = random.randint(osc_freq[0], osc_freq[1])
osc_sine = sim.sim_oscillation(n_seconds, fs, osc_freq_rand, cycle='sine')
plot_time_series(times[plt_time[0]:plt_time[1]], osc_sine[plt_time[0]:plt_time[1]])

# simulate brown noise
# get random exponential 
exp_rand = random.uniform(exp[0], exp[1])
br_noise = sim.sim_powerlaw(n_seconds, fs, exp_rand)

plot_time_series(times[plt_time[0]:plt_time[1]], br_noise[plt_time[0]:plt_time[1]])

#
#freq_range = [4, 150] # for peak detection
#bw_lims = [2, 6]
#max_n_peaks = 5
#            
signal = osc_sine + br_noise
plot_time_series(times[plt_time[0]:plt_time[1]], signal[plt_time[0]:plt_time[1]])
#
#plot_time_series(times[plt_time[0]:plt_time[1]], signal[plt_time[0]:plt_time[1]])
#
#
## compute frequency spectrum
#freq_mean, psd_mean = spectral.compute_spectrum(signal, fs, method='welch', avg_type='mean', nperseg=fs*2)
#
## Initialize FOOOF model
#fm = FOOOF(peak_width_limits=bw_lims, background_mode='knee', max_n_peaks=max_n_peaks)
#
## fit model
#fm.fit(freq_mean, psd_mean, freq_range)    
#
#fm.report()
#

#%% Create sampling distributions

# get right distribution for pi
# set parameters normal distribution with changing SD 
# and bounds from -pi to pi
mean = 0.5 
# sd = [0.04, 0.08, 0.16, 0.32, 0.64] 
sd = 0.08
low_bound = 0
upp_bound = 1

hf_distribution = truncnorm((low_bound - mean) / sd, (upp_bound - mean) / sd, loc=mean, scale=sd)

# get single number from distribution
hf_distribution_array = hf_distribution.rvs(osc_freq_rand * n_seconds)

# calculate number of bursts per cycle - 1000 bursts per second
total_bursts_sec = 10000
bursts_per_cycle = int(total_bursts_sec / osc_freq_rand)

# 100 samples of the distribution for each cycle in signal
hf_distribution_array = [hf_distribution.rvs(bursts_per_cycle) for ii in range(0,osc_freq_rand * n_seconds)]

plt.hist(hf_distribution_array[0])

# for each cycle of simulated data 


# get random number between -pi and pi with that distribution


# 

#%%

# calculate length cycle
samples_cycle = 1/osc_freq_rand * fs

# calculate length signal
signal_length = n_seconds * fs

# create array with sample in which cycle starts
cycle_array = np.linspace(0,signal_length, osc_freq_rand * n_seconds, endpoint=False)


# for each cycle, we calculate the samples in which the samples from distribution occur
hf_activity_array = [np.nan] * len(cycle_array)

for ii in range(len(cycle_array)): 
    
    hf_activity_array[ii] = [
            (cycle_array[ii] + (hf_distribution_array[ii][jj] * samples_cycle))
            for jj in range(len(hf_distribution_array[ii]))]
    
    


#%%

fs = 1000
hf_range = [80,250]
ii=0
# signal of zeros with length cycle
hf_sig = np.zeros([int(np.floor(samples_cycle))])

for kk in range(0,len(hf_activity_array[ii])):
    
    # get random high frequency and length of a cycle of that frequency
    hf_freq = random.randint(hf_range[0],hf_range[1])
    n_seconds_cycle = (1/hf_freq * fs)/fs
    
    # create cycle of high frequency with 0.05 'power'
    hf_cycle = np.sin(2*np.pi*1/n_seconds_cycle * (np.arange(fs*n_seconds_cycle)/fs))*.05
    
    # add cycles to zeros 
    hf_sig[int(round(hf_activity_array[ii][kk]) -  np.ceil(len(hf_cycle) / 2)) : 
        int(round(hf_activity_array[ii][kk]) + np.floor(len(hf_cycle) / 2))] = \
        hf_sig[int(round(hf_activity_array[ii][kk]) -  np.ceil(len(hf_cycle) / 2)) : \
        int(round(hf_activity_array[ii][kk]) + np.floor(len(hf_cycle) / 2))] + hf_cycle
        
plt.plot(hf_sig)
plt.plot(signal[int(np.floor(cycle_array[0])):int(np.floor(cycle_array[1]))])
plt.show()
signal[int(np.floor(cycle_array[0])):int(np.floor(cycle_array[1]))] = \
    signal[int(np.floor(cycle_array[0])):int(np.floor(cycle_array[1]))] + hf_sig
    
plt.plot(signal[int(np.floor(cycle_array[0])):int(np.floor(cycle_array[1]))])



#%% VERSION 2


fs = 1000
n_seconds = 3
osc_freq = [4,12]
plt_time =  [0,3000]
hf_range = [80,250]

exp = [-3, -1]


times = create_times(n_seconds, fs)



#%% create and plot LF oscillation

# simulate oscillation
# get random oscillation between frequency ranges
osc_freq_rand = random.randint(osc_freq[0], osc_freq[1])
osc_sine = sim.sim_oscillation(n_seconds, fs, osc_freq_rand, cycle='sine')
plot_time_series(times[plt_time[0]:plt_time[1]], osc_sine[plt_time[0]:plt_time[1]])

# simulate brown noise
# get random exponential 
exp_rand = random.uniform(exp[0], exp[1])
br_noise = sim.sim_powerlaw(n_seconds, fs, exp_rand)

plot_time_series(times[plt_time[0]:plt_time[1]], br_noise[plt_time[0]:plt_time[1]])

#
#freq_range = [4, 150] # for peak detection
#bw_lims = [2, 6]
#max_n_peaks = 5
#            
signal = osc_sine + br_noise
plot_time_series(times[plt_time[0]:plt_time[1]], signal[plt_time[0]:plt_time[1]])
#
#plot_time_series(times[plt_time[0]:plt_time[1]], signal[plt_time[0]:plt_time[1]])
#
#
## compute frequency spectrum
#freq_mean, psd_mean = spectral.compute_spectrum(signal, fs, method='welch', avg_type='mean', nperseg=fs*2)
#
## Initialize FOOOF model
#fm = FOOOF(peak_width_limits=bw_lims, background_mode='knee', max_n_peaks=max_n_peaks)
#
## fit model
#fm.fit(freq_mean, psd_mean, freq_range)    
#
#fm.report()
#

#%% Create sampling distributions


plt.hist(hf_distribution_array)

# for each cycle of simulated data 


# get random number between -pi and pi with that distribution


# 

#%%

# get right distribution for pi
# set parameters normal distribution with changing SD 
# and bounds from -pi to pi
mean = 0.5 
# sd = [0.04, 0.08, 0.16, 0.32, 0.64] 
sd = 0.08
low_bound = 0
upp_bound = 1

# creat distribution function
hf_distribution = truncnorm((low_bound - mean) / sd, (upp_bound - mean) / sd, loc=mean, scale=sd)

# calculate length cycle
samples_cycle = 1/osc_freq_rand * fs

# calculate length signal
signal_length = n_seconds * fs

# create array with sample in which each cycle starts
cycle_array = np.linspace(0,signal_length, osc_freq_rand * n_seconds, endpoint=False)

# cc = for each cycle 
for cc in range(len(cycle_array)-1):
            
    # calculate number of bursts per cycle - 1000 bursts per second
    total_bursts_sec = 10000
    bursts_per_cycle = int(total_bursts_sec / osc_freq_rand)
    
    # 100 samples of the distribution for each cycle in signal
    hf_distribution_array = hf_distribution.rvs(bursts_per_cycle)
    
    # translate the distributions to each sample were activity is
    hf_activity_array = [(cycle_array[cc] + (hf_distribution_array[ii] * samples_cycle)) 
                        for ii in range(len(hf_distribution_array))]   
    
    
    for kk in range(0,len(hf_activity_array)):
        
        # get random high frequency and length of a cycle of that frequency
        hf_freq = random.randint(hf_range[0],hf_range[1])
        n_seconds_cycle = (1/hf_freq * fs)/fs
        
        # create cycle of high frequency with 0.05 'power'
        hf_cycle = np.sin(2*np.pi*1/n_seconds_cycle * (np.arange(fs*n_seconds_cycle)/fs))*.05
        
        # add cycles to zeros signal - in the center
        signal[int(round(hf_activity_array[kk]) -  np.ceil(len(hf_cycle) / 2)) : 
            int(round(hf_activity_array[kk]) + np.floor(len(hf_cycle) / 2))] = \
            signal[int(round(hf_activity_array[kk]) -  np.ceil(len(hf_cycle) / 2)) : \
            int(round(hf_activity_array[kk]) + np.floor(len(hf_cycle) / 2))] + hf_cycle
#            
#    signal[int(np.floor(cycle_array[cc])):int(np.floor(cycle_array[cc + 1]))] = \
#        signal[int(np.floor(cycle_array[cc])):int(np.floor(cycle_array[cc + 1]))] + hf_sig

plot_time_series(times[plt_time[0]:plt_time[1]], signal[plt_time[0]:plt_time[1]])

#%%
              
plt.plot(hf_sig)
plt.plot(signal[int(np.floor(cycle_array[0])):int(np.floor(cycle_array[1]))])
plt.show()

    
plt.plot(signal[int(np.floor(cycle_array[0])):int(np.floor(cycle_array[1]))])

#%%
plt_time =  [0,50000]
plot_time_series(times[plt_time[0]:plt_time[1]], signal[plt_time[0]:plt_time[1]])


# cycle array is the cc we need to iterate over
# 0 = cc, while 1 = cc +1 
# however this will give problems at the end of the signal
# so only iterate over len(cycle_array) -1 

# might be problems in the first part of signal as well with bigger SD
# adding some 000's and the beginning and end is also possible
# however do this only after defining the cycle_array












#%%
plt.plot(times[0:2000], signal[0:2000])
signal = np.roll(signal, random.randint(0,signal_length))


#%% CURRENTLY PROBLEM WITH FOOOOF - could be a lot of harmonics (max 50)
    # only recognize them if they are a peaks of certain (high) amplitudes
    # otherwise the slope will go down because it recognizes small peaks

#
#for test in range(10): 
#    
#    fs = 1000
#    n_seconds = 32
#    freq_list = [5, 10, 20]
#    plt_time =  [0,3000]
#    hf_range = [80,250]
#    exp = [-2, -1]
#    times = create_times(n_seconds, fs)
#    
#    # create and plot LF oscillation
#    
#    # simulate oscillation
#    # get random oscillation between frequency ranges
#    
#    # osc_freq_rand = random.randint(osc_freq[0], osc_freq[1])
#    osc_freq_rand = random.choice(freq_list)
#    osc_sine = sim.sim_oscillation(n_seconds, fs, osc_freq_rand, cycle='sine')
#    
#    signal = osc_sine
#    
#    
#    # get random exponential 
#    exp_rand = random.uniform(exp[0], exp[1])
#    
#    # simulate 50x brown noise on top to get reliable slope
#    n_brown_noise = 50
#    for br in range(n_brown_noise):
#        br_noise = sim.sim_powerlaw(n_seconds, fs, exp_rand)
#        signal = signal + br_noise
#    
#    freq_range = [4, 250] # for peak detection
#    bw_lims = [1, 2]
#    max_n_peaks = 1
#    
#    # compute frequency spectrum
#    freq_mean, psd_mean = spectral.compute_spectrum(signal, fs, method='welch', avg_type='mean', nperseg=fs*2)
#    
#    # Initialize FOOOF model
#    fm = FOOOF(peak_width_limits=bw_lims, background_mode='fixed', max_n_peaks=max_n_peaks)
#    
#    # fit model
#    fm.fit(freq_mean, psd_mean, freq_range)    
#    
#    fm.report()
#    
#    # get right distribution for pi
#    # set parameters normal distribution with changing SD 
#    # and bounds from -pi to pi
#    mean = 0.5 
#    sd_list = [0.01, 0.04, 0.16, 0.64] 
#    sd = random.choice(sd_list)
#    low_bound = 0
#    upp_bound = 1
#    
#    # creat distribution function
#    hf_distribution = truncnorm((low_bound - mean) / sd, (upp_bound - mean) / sd, loc=mean, scale=sd)
#    
#    # calculate length cycle
#    samples_cycle = 1/osc_freq_rand * fs
#    
#    # calculate length signal
#    signal_length = n_seconds * fs
#    
#    # create array with sample in which each cycle starts
#    cycle_array = np.linspace(0,signal_length, osc_freq_rand * n_seconds, endpoint=False)
#    
#    # calculate number of bursts per cycle - 1000 bursts per second
#    total_bursts_sec = 1000 * n_seconds
#    bursts_per_cycle = int(total_bursts_sec / osc_freq_rand)
#    
#    # cc = for each cycle 
#    for cc in range(1,len(cycle_array)-1):
#                    
#        # 100 samples of the distribution for each cycle in signal
#        hf_distribution_array = hf_distribution.rvs(bursts_per_cycle)
#        
#        # translate the distributions to each sample were activity is
#        hf_activity_array = [(cycle_array[cc] + (hf_distribution_array[ii] * samples_cycle)) 
#                            for ii in range(len(hf_distribution_array))]   
#        
#        # kk = for each simulated high freq activity
#        for kk in range(len(hf_activity_array)):
#            
#            # get random high frequency and length of a cycle of that frequency
#            hf_freq = random.randint(hf_range[0],hf_range[1])
#            n_seconds_cycle = (1/hf_freq * fs)/fs
#            
#            # create cycle of high frequency with 0.05 'power'
#            hf_cycle = np.sin(2*np.pi*1/n_seconds_cycle * (np.arange(fs*n_seconds_cycle)/fs))*.05
#            
#            # add cycles to zeros signal - in the center
#            signal[int(round(hf_activity_array[kk]) -  np.ceil(len(hf_cycle) / 2)) : 
#                int(round(hf_activity_array[kk]) + np.floor(len(hf_cycle) / 2))] = \
#                signal[int(round(hf_activity_array[kk]) -  np.ceil(len(hf_cycle) / 2)) : \
#                int(round(hf_activity_array[kk]) + np.floor(len(hf_cycle) / 2))] + hf_cycle
#    
#        
#    # cut off first second and last second of signal
#    signal = signal[fs*1:-fs*1]
#    
#    freq_range = [4, 250] # for peak detection
#    bw_lims = [1, 2]
#    max_n_peaks = 20
#    
#    # compute frequency spectrum
#    freq_mean, psd_mean = spectral.compute_spectrum(signal, fs, method='welch', avg_type='mean', nperseg=fs*2)
#    
#    # Initialize FOOOF model
#    fm = FOOOF(peak_width_limits=bw_lims, background_mode='fixed', max_n_peaks=max_n_peaks)
#    
#    # fit model
#    fm.fit(freq_mean, psd_mean, freq_range)    
#    
#    fm.report()


#%%

osc_freq_rand = 20
osc_sine = sim.sim_oscillation(n_seconds, fs, osc_freq_rand, cycle='sine')
signal = osc_sine
print(osc_freq_rand)
fs = 1000
f = osc_freq_rand
sample = 30 * fs
x = np.arange(sample)  
y = np.sin(2 * np.pi * f * x / fs)

plt_time =  [29000,30000]

plt_time =  [0,1000]
plt.figure(figsize=(20,10))
plt.plot(times[plt_time[0]:plt_time[1]], y[plt_time[0]:plt_time[1]], label='sinusoid') 
plt.plot(times[plt_time[0]:plt_time[1]], signal[plt_time[0]:plt_time[1]], label = 'neurodsp sinusoid')
plt.legend(fontsize=20)
plt.show()


plt.figure(figsize=(20,10))
plt.plot(times[plt_time[0]:plt_time[1]], y[plt_time[0]:plt_time[1]]) 
plt.plot(times[plt_time[0]:plt_time[1]], signal[plt_time[0]:plt_time[1]])
plt.show()
