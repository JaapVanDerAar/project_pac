#%% Loading necessary data for next steps

import os 
import numpy as np

# change dir 
os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab')

# data
datastruct = np.load('datastruct_fpb.npy', allow_pickle=True)
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
os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab\Code\distinguish_pac')

from bycycle.filt import lowpass_filter
from bycycle.features import compute_features
import matplotlib.pyplot as plt
import module_pac_plots as pac_plt
import module_pac_functions as pacf
from scipy.signal import hilbert

#%% Loop over all channels with PAC using the CF and BW for phase data 
### And extract and save the cycle-by-cycle features

# create empty output
rdsym = []
ptsym = []
bursts = []

#burst_kwargs = {'amplitude_fraction_threshold': 0,
#                'amplitude_consistency_threshold': .2,
#                'period_consistency_threshold': .45,
#                'monotonicity_threshold': .7,
#                'N_cycles_min': 3}
#
#burst_kwargs = {'amplitude_fraction_threshold': 0,
#                'amplitude_consistency_threshold': .25,
#                'period_consistency_threshold': .45,
#                'monotonicity_threshold': .6,
#                'N_cycles_min': 3}

burst_kwargs = {'amplitude_fraction_threshold': 0.25,
                'amplitude_consistency_threshold': .4,
                'period_consistency_threshold': .45,
                'monotonicity_threshold': .6,
                'N_cycles_min': 3}

# for every channel with pac
for ii in range(len(pac_idx[0])):
    
    # get subj & ch
    subj = pac_idx[0][ii]
    ch = pac_idx[1][ii]
    
    if (psd_peaks[subj][ch][0] < 15) & (psd_peaks[subj][ch][1] >.2) & (psd_peaks[subj][ch][1] < 1.5):
        
        # get phase providing band
        lower_phase = psd_peaks[subj][ch][0] - (psd_peaks[subj][ch][2] / 2)
        upper_phase = psd_peaks[subj][ch][0] + (psd_peaks[subj][ch][2] / 2)
        
        fs = 1000
        f_range = [lower_phase, upper_phase]
        f_lowpass = 55
        N_seconds = len(datastruct[subj][ch]) / fs - 2
        
        signal = lowpass_filter(datastruct[subj][ch], fs, f_lowpass, N_seconds=N_seconds, remove_edge_artifacts=False)
        
        df = compute_features(signal, fs, f_range,  burst_detection_kwargs=burst_kwargs)
        
        is_burst = df['is_burst'].tolist()
        time_rdsym = df['time_rdsym'].to_numpy()
        time_ptsym = df['time_ptsym'].to_numpy()
        
        bursts.append(is_burst)
        rdsym.append(time_rdsym)
        ptsym.append(time_ptsym)

        
#%% Save
    
np.save('rdsym', rdsym)
np.save('ptsym', ptsym)
np.save('bursts', bursts)



#%% Or load

mean_time_rdsym = np.load('mean_time_rdsym.npy')   
median_time_rdsym = np.load('median_time_rdsym.npy')   
mean_time_ptsym = np.load('mean_time_ptsym.npy')   
median_time_ptsym = np.load('median_time_ptsym.npy')   
    
#%%
plt.scatter(median_time_ptsym, median_time_rdsym)

plt.scatter(mean_time_ptsym, mean_time_rdsym)
plt.ylim([.49,.525])
plt.xlabel('Peak-Through Sym')
plt.ylabel('Rise-Decay Sym')


#%% Get cycle specific data per channel

ii = 10


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
N_seconds = 2 #??????????????? length signal / fs - 2 


signal = lowpass_filter(datastruct[subj][ch], fs, f_lowpass, remove_edge_artifacts=False)

df = compute_features(signal, fs, f_range)


plt.hist(df.time_rdsym, bins=20)
plt.title('Rise-Decay Sym')
plt.show()


plt.hist(df.time_ptsym, bins=20)
plt.title('Peak-Trough Sym')
plt.show()

plt.scatter(df.time_ptsym, df.time_rdsym)
plt.xlabel('Peak-Trough Sym')
plt.ylabel('Rise-Decay Sym')
plt.title('Two symmetry measures per cycle for a channel')

plt_time = [0, 5]   

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
       
#%% Overview of all CFs

cf = []
bw = []

for subj in range(len(psd_peaks)):
    for ch in range(len(psd_peaks[subj])):
        if len(psd_peaks[subj][ch]) > 0:
            cf.append(psd_peaks[subj][ch][0])
            bw.append(psd_peaks[subj][ch][1])
        
plt.hist(cf, bins=60)
plt.title('distribution of CFs')
plt.show()
plt.hist(bw, bins=20)
plt.title('distribution of BWs')
 
plt.figure(figsize=(10,10))
plt.scatter(bw,cf)
       
#%% Overview of CFs of PAC channels

cf = [] 
bw = []

for ii in range(len(pac_idx[0])):
    # get subj & ch
    subj = pac_idx[0][ii]
    ch = pac_idx[1][ii]
    if len(psd_peaks[subj][ch]) > 0:
        cf.append(psd_peaks[subj][ch][0])
        bw.append(psd_peaks[subj][ch][2])
    
plt.hist(cf, bins=60)
plt.show()
plt.hist(bw, bins=20)

plt.figure(figsize=(10,10))
plt.scatter(bw,cf)


# clean data
# burst detection 
# create preprocessing pipeline up till the point where the features are extracted. 

### Later step: look per cycle and to the direction (pi) of the phase. These change with
### the symmetry of the waveform. Consistent shapes of waveforms might results in PAC. 
### So the consistency of the symmetry should within a channel should be a feature

### FEATURES
### - BW
### - CF
### - AM/power
### - PAC value/ Rho
### - RD Symmetry
### - PT Symmetry
### - RD Symmetry STD
### - PT Symmetry STD

### - periodic and aperiodic measures? 

#%% 
subj = 0 
ch = 0

signal = datastruct[subj][ch]

from bycycle.burst import plot_burst_detect_params

#burst_kwargs = {'amplitude_fraction_threshold': 0,
#                'amplitude_consistency_threshold': .2,
#                'period_consistency_threshold': .45,
#                'monotonicity_threshold': .7,
#                'N_cycles_min': 3}

burst_kwargs = {'amplitude_fraction_threshold': 0,
                'amplitude_consistency_threshold': .2,
                'period_consistency_threshold': .45,
                'monotonicity_threshold': .7,
                'N_cycles_min': 3}

lower_phase = psd_peaks[subj][ch][0] - (psd_peaks[subj][ch][2] / 2)
upper_phase = psd_peaks[subj][ch][0] + (psd_peaks[subj][ch][2] / 2)

Fs = 1000

f_range = [lower_phase, upper_phase]
df = compute_features(signal, Fs, f_range, burst_detection_kwargs=burst_kwargs)

plot_burst_detect_params(signal, Fs, df, burst_kwargs,tlims=None, figsize=(16, 3))

#%%
burst_kwargs = {'amplitude_fraction_threshold': 0,
                'amplitude_consistency_threshold': .2,
                'period_consistency_threshold': .45,
                'monotonicity_threshold': .7,
                'N_cycles_min': 3}

f_range = [lower_phase, upper_phase]
low = int(round(f_range[0]))
up = int(round(f_range[1]))
f_range2 = [low, up]


df = compute_features(signal, fs, f_range2, burst_detection_kwargs=burst_kwargs)

plot_burst_detect_params(signal, Fs, df, burst_kwargs, tlims=None, figsize=(16, 3))

#%% Change datastruct data to float64

for subj in range(len(datastruct)):
    for ch in range(len(datastruct[subj])):
        datastruct[subj][ch] = datastruct[subj][ch].astype(np.float64)

#%% Plot to find right detection
ii = 192
burst_kwargs = {'amplitude_fraction_threshold': 0.25,
                'amplitude_consistency_threshold': .4,
                'period_consistency_threshold': .45,
                'monotonicity_threshold': .6,
                'N_cycles_min': 3}

# get subj & ch
subj = clean_db['subj'][ii]
ch = clean_db['ch'][ii]

# get phase providing band
lower_phase = psd_peaks[subj][ch][0] - (psd_peaks[subj][ch][2] / 2)
upper_phase = psd_peaks[subj][ch][0] + (psd_peaks[subj][ch][2] / 2)

fs = 1000
f_range = [lower_phase, upper_phase]
f_lowpass = 55
N_seconds = len(datastruct[subj][ch]) / fs - 2

#signal = lowpass_filter(datastruct[subj][ch], fs, f_lowpass, N_seconds=N_seconds, remove_edge_artifacts=False)
signal = datastruct[subj][ch]
signal = signal[10000:20000]


df = compute_features(signal, fs, f_range,  burst_detection_kwargs=burst_kwargs)

plot_burst_detect_params(signal, fs, df, burst_kwargs, tlims=None, figsize=(16, 3))

plt.scatter(df['time_ptsym'][(df['is_burst'] == True)],df['time_rdsym'][(df['is_burst'] == True)])
plt.scatter(df['time_ptsym'][(df['is_burst'] == False)],df['time_rdsym'][(df['is_burst'] == False)])
plt.xlabel('PT')
plt.ylabel('RD')
plt.show()


plt.hist(df['time_ptsym'][(df['is_burst'] == True)],alpha=.5)
plt.hist(df['time_ptsym'][(df['is_burst'] == False)],alpha=.5)
plt.show()

plt.hist(df['time_rdsym'][(df['is_burst'] == True)],alpha=.5)
plt.hist(df['time_rdsym'][(df['is_burst'] == False)],alpha=.5)
plt.show()
