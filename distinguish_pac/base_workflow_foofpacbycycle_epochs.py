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

import pandas as pd

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

np.save('datastruct_full', datastruct)
np.save('elec_locs', elec_locs)


#%% Create dataframe that will be used throughout

# df = pd.DataFrame(np.nan, index=range(0,10), columns=['A'])

# find length of subj * ch
size = 0
for subj in range(len(datastruct)): 
    size = size + len(datastruct[subj])
    
epoch_len = 20 # in seconds
num_epochs = int(timewindow/epoch_len)
    
# create dataframe with columns: subj & ch
features_df = pd.DataFrame(index=range(0,size*num_epochs), columns=['subj', 'ch', 'ep'])

# fill subj & ch
features_df['subj'] = [subj for subj in range(len(datastruct)) 
                        for ch in range(len(datastruct[subj]))
                        for ep in range(num_epochs)]
features_df['ch'] = [ch for subj in range(len(datastruct)) 
                        for ch in range(len(datastruct[subj]))
                        for ep in range(num_epochs)]
features_df['ep'] = [ep for subj in range(len(datastruct)) 
                        for ch in range(len(datastruct[subj]))
                        for ep in range(num_epochs)]


#%% Calculate largest peak in PSD using FOOF

# original is [4, 55], [2, 8], 4
freq_range = [4, 55] # for peak detection
bw_lims = [2, 6]
max_n_peaks = 5
freq_range_long = [4, 118] # to establish a more reliable slope

psd_peaks, backgr_params, backgr_params_long = detect_pac.fooof_highest_peak_epoch(datastruct, epoch_len, fs, freq_range, bw_lims, max_n_peaks, freq_range_long)

#%% Write FOOOF features to dataframe

# periodic component
features_df['CF'] = [
        psd_peaks[features_df['subj'][ii]][features_df['ch'][ii]][features_df['ep'][ii]][0]
        for ii in range(len(features_df))
        ]
features_df['Amp'] = [
        psd_peaks[features_df['subj'][ii]][features_df['ch'][ii]][features_df['ep'][ii]][1] 
        for ii in range(len(features_df))
        ]
features_df['BW'] = [
        psd_peaks[features_df['subj'][ii]][features_df['ch'][ii]][features_df['ep'][ii]][2] 
        for ii in range(len(features_df))
        ]

# aperiodic component
features_df['offset'] = [
        backgr_params[features_df['subj'][ii]][features_df['ch'][ii]][features_df['ep'][ii]][0] 
        for ii in range(len(features_df))
        ]
features_df['knee'] = [
        backgr_params[features_df['subj'][ii]][features_df['ch'][ii]][features_df['ep'][ii]][1] 
        for ii in range(len(features_df))
        ]
features_df['exp'] = [
        backgr_params[features_df['subj'][ii]][features_df['ch'][ii]][features_df['ep'][ii]][2] 
        for ii in range(len(features_df))
        ]

# aperiodic component long frequency range
features_df['offset_long'] = [
        backgr_params_long[features_df['subj'][ii]][features_df['ch'][ii]][features_df['ep'][ii]][0] 
        for ii in range(len(features_df))
        ]
features_df['knee_long'] = [
        backgr_params_long[features_df['subj'][ii]][features_df['ch'][ii]][features_df['ep'][ii]][1] 
        for ii in range(len(features_df))
        ]
features_df['exp_long'] = [
        backgr_params_long[features_df['subj'][ii]][features_df['ch'][ii]][features_df['ep'][ii]][2] 
        for ii in range(len(features_df))
        ]


#%% So now we have the peaks. Use those as input phase frequency for detecting PAC
    
amplitude_providing_band = [62, 118]; #80-125 Hz band as original

features_df = detect_pac.cal_pac_values_varphase(datastruct, amplitude_providing_band, fs, features_df, epoch_len)

#%% How many channels have PAC when channels are NOT resampled?

perc = features_df['pac_presence'][features_df['pac_presence'] == 1].sum()/ \
        len(features_df['pac_presence'])

print('in ', perc * 100, 'channels is PAC (unresampled)')


#%% Run resampling with variable phase and calculate true Z-values of PACs

# to ensure it does not give warnings when we change a specific value
# in a specific column 
pd.options.mode.chained_assignment = None

num_resamples = 1000

features_df = detect_pac.resampled_pac_varphase(datastruct, amplitude_providing_band, fs, num_resamples, features_df, epoch_len)

features_df.to_csv('features_df_epoch.csv', sep=',', index=False)


#%% How many channels have PAC after resampling?

perc_resamp = features_df['resamp_pac_presence'][features_df['resamp_pac_presence'] == 1].sum()/ \
        len(features_df['resamp_pac_presence'])

print('in ', perc_resamp * 100, 'channels is PAC (resampled)')

# pac_idx = list(np.where(features_df['resamp_pac_presence'] == 1))
    
#%% Load all data that where created above to start ByCycle

# change dir 
os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab')

# data
datastruct = np.load('datastruct_full.npy', allow_pickle=True)
elec_locs = np.load('elec_locs.npy', allow_pickle=True)

# dataframe
features_df = pd.read_csv('features_df.csv', sep=',')

#%% Get bycycle features STILL SOME ATTENUATION PROBLEMS AND WRITE INTO FUNCTION

f_lowpass = 55
N_seconds = timewindow - 2

burst_kwargs = {'amplitude_fraction_threshold': 0.25,
                'amplitude_consistency_threshold': .4,
                'period_consistency_threshold': .45,
                'monotonicity_threshold': .6,
                'N_cycles_min': 3}

features_df['volt_amp'] = np.nan
features_df['rdsym']  = np.nan
features_df['ptsym']  = np.nan

# for every channel with pac
for ii in range(len(features_df)):
    
    # for every channel that has peaks
    if ~np.isnan(features_df['CF'][ii]):
    
        # define phase providing band
        CF = features_df['CF'][ii]
        BW = features_df['BW'][ii]
        
        phase_providing_band= [(CF - (BW/2)),  (CF + (BW/2))]
        
        subj = features_df['subj'][ii]
        ch = features_df['ch'][ii]
        data = datastruct[subj][ch] 
        
        signal = lowpass_filter(data, fs, f_lowpass, N_seconds=N_seconds, remove_edge_artifacts=False)
        
        bycycle_df = compute_features(signal, fs, phase_providing_band,  burst_detection_kwargs=burst_kwargs)
        
        features_df['volt_amp'][ii] = bycycle_df['volt_peak'].median()
        features_df['rdsym'][ii] = bycycle_df['time_rdsym'].median()
        features_df['ptsym'][ii] = bycycle_df['time_ptsym'].median()
        
        print('this was ch', ii)
        
        
features_df.to_csv('features_df.csv', sep=',', index=False)


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
