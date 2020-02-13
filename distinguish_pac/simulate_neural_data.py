from neurodsp import sim, spectral
from neurodsp.utils import create_times

from neurodsp.plts.spectral import plot_power_spectra
from neurodsp.plts.time_series import plot_time_series

from fooof import FOOOF
from bycycle.filt import lowpass_filter
from bycycle.features import compute_features
import random
from scipy.stats import truncnorm
import numpy as np
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
n_seconds = 100
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

# get right distribution for pi
# set parameters normal distribution with changing SD 
# and bounds from -pi to pi
mean = 0.5 
# sd = [0.04, 0.08, 0.16, 0.32, 0.64] 
sd = 0.08
low_bound = 0
upp_bound = 1

hf_distribution = truncnorm((low_bound - mean) / sd, (upp_bound - mean) / sd, loc=mean, scale=sd)


# calculate number of bursts per cycle - 1000 bursts per second
total_bursts_sec = 10000
bursts_per_cycle = int(total_bursts_sec / osc_freq_rand)

# 100 samples of the distribution for each cycle in signal
hf_distribution_array = hf_distribution.rvs(bursts_per_cycle)

plt.hist(hf_distribution_array)

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

# cc = cycle 
cc = 0
hf_activity_array = [(cycle_array[cc] + (hf_distribution_array[ii] * samples_cycle)) 
                    for ii in range(len(hf_distribution_array))]

    
    


#%%

# signal of zeros with length cycle
hf_sig = np.zeros([int(np.floor(samples_cycle))])

for kk in range(0,len(hf_activity_array)):
    
    # get random high frequency and length of a cycle of that frequency
    hf_freq = random.randint(hf_range[0],hf_range[1])
    n_seconds_cycle = (1/hf_freq * fs)/fs
    
    # create cycle of high frequency with 0.05 'power'
    hf_cycle = np.sin(2*np.pi*1/n_seconds_cycle * (np.arange(fs*n_seconds_cycle)/fs))*.05
    
    # add cycles to zeros 
    hf_sig[int(round(hf_activity_array[kk]) -  np.ceil(len(hf_cycle) / 2)) : 
        int(round(hf_activity_array[kk]) + np.floor(len(hf_cycle) / 2))] = \
        hf_sig[int(round(hf_activity_array[kk]) -  np.ceil(len(hf_cycle) / 2)) : \
        int(round(hf_activity_array[kk]) + np.floor(len(hf_cycle) / 2))] + hf_cycle
        
plt.plot(hf_sig)
plt.plot(signal[int(np.floor(cycle_array[0])):int(np.floor(cycle_array[1]))])
plt.show()
signal[int(np.floor(cycle_array[0])):int(np.floor(cycle_array[1]))] = \
    signal[int(np.floor(cycle_array[0])):int(np.floor(cycle_array[1]))] + hf_sig
    
plt.plot(signal[int(np.floor(cycle_array[0])):int(np.floor(cycle_array[1]))])



# cycle array is the cc we need to iterate over
# 0 = cc, while 1 = cc +1 
# however this will give problems at the end of the signal
# so only iterate over len(cycle_array) -1 

# might be problems in the first part of signal as well with bigger SD
# adding some 000's and the beginning and end is also possible
# however do this only after defining the cycle_array





