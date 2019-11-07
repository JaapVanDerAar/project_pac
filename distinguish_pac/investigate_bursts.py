import os 
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn import linear_model
import pandas as pd

#%%  Load data 

# change dir 
os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab')

load_data = open('clean_db.pkl','rb')
clean_db = pickle.load(load_data)

# Extract data from dictionary
subj_name   = clean_db['subj_name']
subj        = clean_db['subj']
ch          = clean_db['ch']
locs        = clean_db['locs']
dat_name    = clean_db['dat_name']
data        = clean_db['data']
pac_rhos    = clean_db['pac_rhos']
resamp_zvals = clean_db['resamp_zvals']
resamp_pvals = clean_db['resamp_pvals']
psd_params  = clean_db['psd_params']
rd_sym      = clean_db['rd_sym']
pt_sym      = clean_db['pt_sym']
bursts      = clean_db['bursts']
fs          = clean_db['fs']

# only take data with bursts
pt_sym_bursts = []
pt_sym_nobursts = []

rd_sym_bursts = []
rd_sym_nobursts = []

for ii in range(len(rd_sym)):
    
    pt_yes_burst = [pt_sym[ii][jj] for jj in range(len(pt_sym[ii])) if bursts[ii][jj] == True]
    pt_no_burst = [pt_sym[ii][jj] for jj in range(len(pt_sym[ii])) if bursts[ii][jj] == False]
    
    pt_sym_bursts.append(pt_yes_burst)
    pt_sym_nobursts.append(pt_no_burst)    
    
    rd_yes_burst = [rd_sym[ii][jj] for jj in range(len(rd_sym[ii])) if bursts[ii][jj] == True]
    rd_no_burst = [rd_sym[ii][jj] for jj in range(len(rd_sym[ii])) if bursts[ii][jj] == False]
    
    rd_sym_bursts.append(rd_yes_burst)
    rd_sym_nobursts.append(rd_no_burst)
    
# calculate other variables 
mean_pt_sym     = [np.mean(pt_sym_bursts[ii]) for ii in range(len(pt_sym_bursts))]
median_pt_sym   = [np.median(pt_sym_bursts[ii]) for ii in range(len(pt_sym_bursts))]
std_pt_sym      = [np.std(pt_sym_bursts[ii]) for ii in range(len(pt_sym_bursts))]

mean_rd_sym     = [np.mean(rd_sym_bursts[ii]) for ii in range(len(rd_sym_bursts))]
median_rd_sym   = [np.median(rd_sym_bursts[ii]) for ii in range(len(rd_sym_bursts))]
std_rd_sym      = [np.std(rd_sym_bursts[ii]) for ii in range(len(rd_sym_bursts))]

psd_cf          = [psd_params[ii][0] for ii in range(len(psd_params))]
psd_amp         = [psd_params[ii][1] for ii in range(len(psd_params))]
psd_bw          = [psd_params[ii][2] for ii in range(len(psd_params))]

#%% Look to what effect bursts or not have on the symmetry measures
### First, create new variables, RD with and without bursts, PT with and without bursts

pt_sym_bursts = []
pt_sym_nobursts = []

rd_sym_bursts = []
rd_sym_nobursts = []

for ii in range(len(rd_sym)):
    
    pt_yes_burst = [pt_sym[ii][jj] for jj in range(len(pt_sym[ii])) if bursts[ii][jj] == True]
    pt_no_burst = [pt_sym[ii][jj] for jj in range(len(pt_sym[ii])) if bursts[ii][jj] == False]
    
    pt_sym_bursts.append(pt_yes_burst)
    pt_sym_nobursts.append(pt_no_burst)    
    
    rd_yes_burst = [rd_sym[ii][jj] for jj in range(len(rd_sym[ii])) if bursts[ii][jj] == True]
    rd_no_burst = [rd_sym[ii][jj] for jj in range(len(rd_sym[ii])) if bursts[ii][jj] == False]
    
    rd_sym_bursts.append(rd_yes_burst)
    rd_sym_nobursts.append(rd_no_burst)
    
#%% Plot some channel specific stuff to get idea of differences
### The no bursts removes the artifical RDs of 0.0 and 1.0 but no further difference
ii = 192
plt.scatter(pt_sym_bursts[ii], rd_sym_bursts[ii],alpha=.4)
plt.scatter(pt_sym_nobursts[ii], rd_sym_nobursts[ii],alpha=.4)
plt.xlabel('PT')
plt.ylabel('RD')
plt.title('Not so much difference between bursts and non-bursts, except for the RD artifacts')
plt.show()

plt.hist(pt_sym_bursts[ii],alpha=.5)
plt.hist(pt_sym_nobursts[ii],alpha=.5)
plt.show()

plt.hist(rd_sym_bursts[ii],alpha=.5)
plt.hist(rd_sym_nobursts[ii],alpha=.5)
plt.show()    





