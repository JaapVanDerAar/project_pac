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

timewindow = 110000/fs # 110800 is the length of shortest recording

#%% Run load data middle timewindow function
   
datastruct, elec_locs = load_data.load_data_timewindow(dat_name, subjects, fs, timewindow)


#%% Save structure

# np.save('datastruct_fpb', datastruct)

#%% Calculate largest peak in PSD using FOOF

psd_peaks, backgr_params = detect_pac.fooof_highest_peak(datastruct, fs)


#%% So now we have the peaks. Use those as input phase frequency for detecting PAC
    
amplitude_providing_band = [80, 125]; #80-125 Hz band

pac_presence, pac_pvals, pac_rhos = detect_pac.cal_pac_values_varphase(datastruct, amplitude_providing_band, fs, psd_peaks)

#%% How many channels have PAC when channels are NOT resampled?

array_size = np.size(pac_presence) 
nan_count = np.isnan(pac_presence).sum().sum()
pp = (pac_presence == 1).sum().sum()
percentage_pac = pp / (array_size - nan_count) * 100
print('in ', percentage_pac, 'channels is PAC (unresampled)')


#%% Run resampling with variable phase and for full time frame of data


num_resamples = 1000

resamp_rho_varphase, resamp_p_varphase = detect_pac.resampled_pac_varphase(datastruct, amplitude_providing_band, fs, num_resamples, psd_peaks)


#%% Calculate true PAC values (by comparing rho value to resampled rho values, 
#   and assuming normal distribution, which seems like that)

pac_true_zvals, pac_true_pvals, pac_true_presence = detect_pac.true_pac_values(pac_rhos, resamp_rho_varphase)

pac_idx = list(np.where(pac_true_presence == 1))


#%%

# channel with PAC to plot
ii = 1

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


pac_plt.plot_signal(datastruct, phase_providing_band, amplitude_providing_band, subj, ch, fs)
    
#%% Load all data that where created above to start ByCycle

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
pac_rhos = np.load('pac_rhos.npy')

# FOOOF features
psd_peaks = np.load('psd_peaks.npy', allow_pickle=True)
backgr_params = np.load('backgr_params.npy', allow_pickle=True)



#%% Now, we only select those channels that have a CF of < 15 Hz, 
### and with an amplitude between .2 & 1.5
### call these subj_idx and ch_idx and from now on DONT use pac_idx!

subj_idx = [pac_idx[0][ii]  \
            for ii in range(len(pac_idx[0]))  \
            if ((psd_peaks[pac_idx[0][ii]][pac_idx[1][ii]][0] < 15) &  \
                (psd_peaks[pac_idx[0][ii]][pac_idx[1][ii]][1] >.2) &  \
                (psd_peaks[pac_idx[0][ii]][pac_idx[1][ii]][1] < 1.5))]
ch_idx = [pac_idx[1][ii]  \
            for ii in range(len(pac_idx[1]))  \
            if ((psd_peaks[pac_idx[0][ii]][pac_idx[1][ii]][0] < 15) &  \
                (psd_peaks[pac_idx[0][ii]][pac_idx[1][ii]][1] >.2) &  \
                (psd_peaks[pac_idx[0][ii]][pac_idx[1][ii]][1] < 1.5))]


#%% Loop over all channels with a CF of < 15 Hz, and with an AMP between .2 & 1.5
### And extract and save the cycle-by-cycle features

# create empty output
rdsym = []
ptsym = []
bursts = []
period = []
volt_amp = []


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
for ii in range(len(subj_idx)):
    
    # get subj & ch
    subj = subj_idx[ii]
    ch = ch_idx[ii]
      
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
    period_ch = df['period'].to_numpy()
    volt_amp_ch = df['volt_amp'].to_numpy()
    
    bursts.append(is_burst)
    rdsym.append(time_rdsym)
    ptsym.append(time_ptsym)
    period.append(period_ch)
    volt_amp.append(volt_amp_ch)

#%% Load save ByCycle measures + index after picking channels with specific CF and Amp
        
#rdsym = np.load('rdsym.npy', allow_pickle=True)  
#ptsym = np.load('ptsym.npy', allow_pickle=True)  
#bursts = np.load('bursts.npy', allow_pickle=True)
#subj_idx = np.load('subj_idx.npy', allow_pickle=True)
#ch_idx = np.load('ch_idx.npy', allow_pickle=True)
##   
#np.save('rdsym', rdsym)
#np.save('ptsym', ptsym)
#np.save('bursts', bursts)   
#np.save('subj_idx', subj_idx)     
#np.save('ch_idx', ch_idx)     
    
#%% Also clean other data by using subj_idx and ch_idx

clean_data = []
clean_pac_rhos = []
clean_resamp_zvals = []
clean_resamp_pvals = []
clean_psd_params = []
clean_backgr_params = []


for ii in range(len(subj_idx)):
    
    subj = subj_idx[ii]
    ch = ch_idx[ii]
    
    clean_data.append(datastruct[subj][ch])
    clean_pac_rhos.append(pac_rhos[subj][ch])
    clean_resamp_zvals.append(pac_true_zvals[subj][ch])
    clean_resamp_pvals.append(pac_true_pvals[subj][ch]) 
    clean_psd_params.append(psd_peaks[subj][ch])
    clean_backgr_params.append(backgr_params[subj][ch])
    
#%% Create database in dictonairy form
    
clean_db = {}

# metadata
clean_db['subj_name'] = subjects # list of initials of subjects (subjects)
clean_db['subj'] = subj_idx # array of the subjects to which data belong (pac_idx[0])
clean_db['ch'] = ch_idx # array of the channels to which data belong (pac_idx[1])
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

save_data = open("clean_db.pkl","wb")
pickle.dump(clean_db,save_data)
save_data.close()


#%% Select all channels with a true oscillation < 15 Hz but 
### without PAC after resampling

pac_idx_nopac = list(np.where(pac_true_presence == 0))

subj_idx_nopac = []
ch_idx_nopac = []

for ii in range(len(pac_idx_nopac[0])):
    if len(psd_peaks[pac_idx_nopac[0][ii]][pac_idx_nopac[1][ii]]) > 0:
        if ((psd_peaks[pac_idx_nopac[0][ii]][pac_idx_nopac[1][ii]][0] < 15) &  \
            (psd_peaks[pac_idx_nopac[0][ii]][pac_idx_nopac[1][ii]][1] >.2) &  \
            (psd_peaks[pac_idx_nopac[0][ii]][pac_idx_nopac[1][ii]][1] < 1.5)): 
            
            subj_idx_nopac.append(pac_idx_nopac[0][ii])
            ch_idx_nopac.append(pac_idx_nopac[1][ii])

            
    
#%%
            
# create empty output
rdsym = []
ptsym = []
bursts = []
period = []
volt_amp = []


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
for ii in range(len(subj_idx_nopac)):
    
    # get subj & ch
    subj = subj_idx_nopac[ii]
    ch = ch_idx_nopac[ii]
      
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


for ii in range(len(subj_idx_nopac)):
    
    subj = subj_idx_nopac[ii]
    ch = ch_idx_nopac[ii]
    
    clean_data.append(datastruct[subj][ch])
    clean_pac_rhos.append(pac_rhos[subj][ch])
    clean_resamp_zvals.append(pac_true_zvals[subj][ch])
    clean_resamp_pvals.append(pac_true_pvals[subj][ch]) 
    clean_psd_params.append(psd_peaks[subj][ch])
    clean_backgr_params.append(backgr_params[subj][ch])

#%% Save under clean_db_nopac    
clean_db_nopac = {}

# metadata
clean_db_nopac['subj_name'] = subjects # list of initials of subjects (subjects)
clean_db_nopac['subj'] = subj_idx_nopac # array of the subjects to which data belong (pac_idx[0])
clean_db_nopac['ch'] = ch_idx_nopac # array of the channels to which data belong (pac_idx[1])
clean_db_nopac['locs'] = elec_locs
clean_db_nopac['dat_name'] = 'fixation_pwrlaw'
clean_db_nopac['fs'] = 1000

# data
clean_db_nopac['data'] = clean_data# list of arrays of channels with PAC

# analysis 
clean_db_nopac['pac_rhos'] = clean_pac_rhos # now array, put in array with only sig PAC chs
clean_db_nopac['resamp_zvals'] = clean_resamp_zvals # is now matrix, put in array with only sig PAC chs
clean_db_nopac['resamp_pvals'] = clean_resamp_pvals # is now matrix, put in array with only sig PAC chs

clean_db_nopac['psd_params'] = clean_psd_params # list of arrays,  put in array with only sig PAC chs
clean_db_nopac['backgr_params'] = clean_backgr_params # list of arrays,  put in array with only sig PAC chs
clean_db_nopac['rd_sym'] = rdsym # list of arrays,  put in array with only sig PAC chs
clean_db_nopac['pt_sym'] = ptsym # list of arrays,  put in array with only sig PAC chs
clean_db_nopac['bursts'] = bursts # list of arrays,  put in array with only sig PAC chs
clean_db_nopac['period'] = period # list of arrays,  put in array with only sig PAC chs
clean_db_nopac['volt_amp'] = volt_amp # list of arrays,  put in array with only sig PAC chs

# Save with pickle
import pickle

save_data = open("clean_db_nopac.pkl","wb")
pickle.dump(clean_db_nopac,save_data)
save_data.close()
