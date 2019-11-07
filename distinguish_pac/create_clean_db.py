
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

# raw rho values
pac_rhos = np.load('pac_rhos.npy')

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

# ByCycle measures
rdsym = np.load('rdsym.npy', allow_pickle=True)  
ptsym = np.load('ptsym.npy', allow_pickle=True)  
bursts = np.load('bursts.npy', allow_pickle=True)
# + full symmetry measures

#%% Only get the data with sign. PAC with CF < 35 Hz and Ampl < 1.5

subj_idx = []
ch_idx = [] 

for ii in range(len(pac_idx[0])):
    subj = pac_idx[0][ii]
    ch = pac_idx[1][ii]
    
    # 
    if (psd_peaks[subj][ch][0] < 15) & (psd_peaks[subj][ch][1] > .2):
        subj_idx.append(subj)
        ch_idx.append(ch)
        #plt.scatter(psd_peaks[subj][ch][0],psd_peaks[subj][ch][1])
        
#%%

#%% Use these clean indexes to select the data from the bigger data to get
### Only the data of the sign. channels

clean_data = []
clean_pac_rhos = []
clean_resamp_zvals = []
clean_resamp_pvals = []
clean_psd_params = []
clean_rdsym = []
clean_ptsym = []
clean_bursts = []


for ii in range(len(subj_idx)):
    
    subj = subj_idx[ii]
    ch = ch_idx[ii]
    
    clean_data.append(datastruct[subj][ch])
    clean_pac_rhos.append(pac_rhos[subj][ch])
    clean_resamp_zvals.append(pac_true_zvals[subj][ch])
    clean_resamp_pvals.append(pac_true_pvals[subj][ch]) 
    clean_psd_params.append(psd_peaks[subj][ch])
    clean_rdsym.append(rdsym[ii])
    clean_ptsym.append(ptsym[ii])
    clean_bursts.append(bursts[ii])
    
    
#%% 
# create database
clean_db = {}

# metedata
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
clean_db['rd_sym'] = clean_rdsym # list of arrays,  put in array with only sig PAC chs
clean_db['pt_sym'] = clean_ptsym # list of arrays,  put in array with only sig PAC chs
clean_db['bursts'] = clean_bursts # list of arrays,  put in array with only sig PAC chs

#%% Save with pickle

import pickle

save_data = open("clean_db.pkl","wb")
pickle.dump(clean_db,save_data)
save_data.close()

#%% Load with Pickle

import pickle
load_data = open('clean_db.pkl','rb')
clean_db = pickle.load(load_data)

