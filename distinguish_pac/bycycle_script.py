#%% Loading necessary data for next steps

# change dir 
os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab')

# data
datastruct = np.load('datastruct.npy', allow_pickle=True)
elec_locs = np.load('elec_locs.npy', allow_pickle=True)

subjects = ['al','ca','cc','de','fp','gc','gf','gw',
          'h0','hh','jc','jm','jp','mv','rh','rr',
          'ug','wc','wm','zt']

# resampled rho and p-values
resamp_rho_varphase = np.load('resamp_rho_varphase.npy', allow_pickle=True) 
resamp_p_varphase = np.load('resamp_p_varphase.npy', allow_pickle=True) 

# resampled statistics    
pac_true_zvals = np.load('pac_true_zvals.npy')
pac_true_pvals = np.load('pac_true_pvals.npy')
pac_true_presence = np.load('pac_true_presence.npy')
pac_idx = np.load('pac_idx.npy')

# psd peaks
psd_peaks = np.load('psd_peaks.npy', allow_pickle=True)

#%% Play around with bycycle

from bycycle.filt import lowpass_filter
from bycycle.features import compute_features
from matplotlib.pyplot import plt
#%%
signal = datastruct[0][10]

fs = 1000
f_lowpass = 100
N_seconds = len(datastruct[0][10]) / fs - 2

signal = lowpass_filter(signal, fs, f_lowpass, N_seconds=N_seconds, remove_edge_artifacts=False)
#%%
f_range = (11, 18)
df = compute_features(signal, fs, f_range)

#%% 

# channel with PAC to plot
ii = 3

# subj & ch
subj = pac_idx[0][ii]
ch = pac_idx[1][ii]

# compute phase band
lower_phase = psd_peaks[subj][ch][0] - (psd_peaks[subj][ch][2] / 2)
upper_phase = psd_peaks[subj][ch][0] + (psd_peaks[subj][ch][2] / 2)

# parameters
fs = 1000
phase_providing_band = [lower_phase, upper_phase]; #4-8 Hz band
amplitude_providing_band = [80, 125]; #80-125 Hz band

#%% 
pac_plt.plot_signal(datastruct, phase_providing_band, amplitude_providing_band, subj, ch, fs)


#%% Loop to get overall channel data

# create empty output
median_time_decay = []
mean_time_decay = []

median_time_rise = []
mean_time_rise = []

median_volt_decay = []
mean_volt_decay = []

median_volt_rise = []
mean_volt_rise = []

median_volt_amp = []
mean_volt_amp = []

median_time_rdsym = []
mean_time_rdsym = []

median_time_ptsym = []
mean_time_ptsym = []

# for every channel with pac
for ii in range(len(pac_idx[0])):
    
    # get subj & ch
    subj = pac_idx[0][ii]
    ch = pac_idx[1][ii]
    
    # get phase providing band
    lower_phase = psd_peaks[subj][ch][0] - (psd_peaks[subj][ch][2] / 2)
    upper_phase = psd_peaks[subj][ch][0] + (psd_peaks[subj][ch][2] / 2)
    
    fs = 1000
    f_range = [lower_phase, upper_phase]
    f_lowpass = 55
    N_seconds = len(datastruct[0][10]) / fs - 2
    
    signal = lowpass_filter(signal, fs, f_lowpass, N_seconds=N_seconds, remove_edge_artifacts=False)
    
    df = compute_features(signal, fs, f_range)

    mean_time_decay.append(df.time_decay.mean())
    median_time_decay.append(df.time_decay.median())
    
    mean_time_rise.append(df.time_rise.mean())
    median_time_rise.append(df.time_rise.median())
    
    mean_volt_decay.append(df.volt_decay.mean())
    median_volt_decay.append(df.volt_decay.median())
    
    mean_volt_rise.append(df.volt_rise.mean())
    median_volt_rise.append(df.volt_rise.median())
    
    mean_volt_amp.append(df.volt_amp.mean())
    median_volt_amp.append(df.volt_amp.median())
    
    mean_time_rdsym.append(df.time_rdsym.mean())
    median_time_rdsym.append(df.time_rdsym.median())
    
    mean_time_ptsym.append(df.time_ptsym.mean())
    median_time_ptsym.append(df.time_ptsym.median())
    
#%% Save
    
np.save('mean_time_rdsym', mean_time_rdsym)
np.save('median_time_rdsym', median_time_rdsym)
np.save('mean_time_ptsym', mean_time_ptsym)
np.save('median_time_ptsym', median_time_ptsym)
    
    
#%%
plt.scatter(median_time_ptsym, median_time_rdsym)


plt.scatter(mean_time_ptsym, mean_time_rdsym)
plt.ylim([.49,.525])

#%% Get cycle specific data per channel

ii = 3


 # get subj & ch
subj = pac_idx[0][ii]
ch = pac_idx[1][ii]

# get phase providing band
lower_phase = psd_peaks[subj][ch][0] - (psd_peaks[subj][ch][2] / 2)
upper_phase = psd_peaks[subj][ch][0] + (psd_peaks[subj][ch][2] / 2)

fs = 1000
f_range = [lower_phase, upper_phase]
phase_providing_band = f_range
f_lowpass = 55
N_seconds = len(datastruct[0][10]) / fs - 2

signal = lowpass_filter(signal, fs, f_lowpass, N_seconds=N_seconds, remove_edge_artifacts=False)

df = compute_features(signal, fs, f_range)


plt.hist(df.time_rdsym, bins=20)
plt.show()


plt.hist(df.time_ptsym, bins=20)
plt.show()

plt.scatter(df.time_ptsym, df.time_rdsym)

plt_time = [0, 2]   

#calculating phase of theta
phase_data = pacf.butter_bandpass_filter(datastruct[subj][ch], phase_providing_band[0], phase_providing_band[1], round(float(fs)));
phase_data_hilbert = hilbert(phase_data);

# filter raw data 
raw_filt = lowpass_filter(datastruct[subj][ch], fs, f_lowpass, N_seconds=N_seconds, remove_edge_artifacts=False)

##calculating amplitude envelope of high gamma
#amp_data = pacf.butter_bandpass_filter(datastruct[subj][ch], amplitude_providing_band[0], amplitude_providing_band[1], round(float(fs)));
#amp_data_hilbert = hilbert(amp_data);

plt.figure(figsize = (20,8));
plt.plot((raw_filt[plt_time[0]*fs:plt_time[1]*fs]),label= 'Raw Signal')
plt.plot((phase_data_hilbert[plt_time[0]*fs:plt_time[1]*fs]),label= 'Phase [{0:.2f} - {1:.2f} Hz]'.format(phase_providing_band[0], phase_providing_band[1]))
#plt.plot((amp_data_hilbert[plt_time[0]*fs:plt_time[1]*fs]),label= 'High Gamma [80-125 Hz]')
plt.scatter([df.loc[0:30].sample_peak], [df.loc[0:30].volt_peak])

plt.xlabel('subj: {0:.0f}, ch {1:.0f},  Two Seconds of Theta Phase, High Gamma Amplitude, and Raw Signal '.format(subj,ch))
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
       


