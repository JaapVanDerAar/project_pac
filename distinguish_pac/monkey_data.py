import os
import numpy as np
import scipy.io as sio


import glob
import h5py
import pandas as pd
import matplotlib.pyplot as plt
#%% Change directory to import necessary modules

os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab\Code\distinguish_pac')

#%% import self-defined modules

import module_load_data as load_data
import module_pac_functions as pacf
import module_detect_pac as detect_pac
import module_pac_plots as pac_plt

from bycycle.filt import lowpass_filter
from bycycle.features import compute_features


#%%

os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab') 
np.save('Vt_data', data_array)    

bt_data = np.load('Vt_data.npy', allow_pickle=True)


#%% meta data

subjects = ['Bt']

channels = [128]

os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab\neurotycho\Bt')

#%% Load data - only 100 trials per channel for now due to memory problems

# which trials
start_tr = 500
end_tr = 600

# initialize datastruct
datastruct = [None] * len(glob.glob("ECoG_ch*.mat")) 

# create channel array
ch_array = np.repeat(range(len(glob.glob("ECoG_ch*.mat"))), len(range(start_tr, end_tr)))

# counter
counter = 0 

# iterate over all ECoG_ch files in map and open and save data in datastruct
for file_idx in glob.glob("ECoG_ch*.mat"):
    ch = file_idx.split('_ch')[1].split('.mat')[0]
    if ch.startswith('0'):
        ch = ch.split('0')[1]


    file = h5py.File(file_idx, 'r')
    ch_dat = np.array(file['data'])

    # get specific part of data of each channel  
    datastruct[counter] = [ch_dat[ch,:] for ch in range(start_tr, end_tr)]
  
    counter = counter + 1
    
# save    
os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab') 
np.save('bt_data', datastruct)    


#%% Load saved data
    
os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab') 
datastruct = np.load('bt_data.npy', allow_pickle=True)  

#%% 



