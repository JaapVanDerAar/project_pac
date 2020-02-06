from neurodsp import sim, spectral
from neurodsp.utils import create_times

from neurodsp.plts.spectral import plot_power_spectra
from neurodsp.plts.time_series import plot_time_series

from fooof import FOOOF
from bycycle.filt import lowpass_filter
from bycycle.features import compute_features

#%% Basic parameters

fs = 1000
n_seconds = 100
osc_freq = 8
plt_time =  [0,3000]
times = create_times(n_seconds, fs)

#%% create and plot LF oscillation
osc_sine = sim.sim_oscillation(n_seconds, fs, osc_freq, cycle='sine')
plot_time_series(times[plt_time[0]:plt_time[1]], osc_sine[plt_time[0]:plt_time[1]])


#%% create and plot bursts

enter_burst = .1
leave_burst = .1

osc_burst = sim.sim_bursty_oscillation(n_seconds, fs, osc_freq,
                                 enter_burst=enter_burst,
                                 leave_burst=leave_burst)
plot_time_series(times[plt_time[0]:plt_time[1]], osc_burst[plt_time[0]:plt_time[1]])


#%% create brown 1/f noise

exponent = -2

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
            
signal = syn_current + osc_burst

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
