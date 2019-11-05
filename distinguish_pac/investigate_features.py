import os 
import numpy as np
import pickle
import matplotlib.pyplot as plt

#%%  Load data 

# change dir 
os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab')

load_data = open('clean_db.pkl','rb')
clean_db = pickle.load(load_data)


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

mean_rd_sym     = [np.mean(rd_sym[ii]) for ii in range(len(rd_sym))]
median_rd_sym   = [np.median(rd_sym[ii]) for ii in range(len(rd_sym))]
std_rd_sym      = [np.std(rd_sym[ii]) for ii in range(len(rd_sym))]

mean_pt_sym     = [np.mean(pt_sym[ii]) for ii in range(len(pt_sym))]
median_pt_sym   = [np.median(pt_sym[ii]) for ii in range(len(pt_sym))]
std_pt_sym      = [np.std(pt_sym[ii]) for ii in range(len(pt_sym))]

psd_cf          = [psd_params[ii][0] for ii in range(len(psd_params))]
psd_amp         = [psd_params[ii][1] for ii in range(len(psd_params))]
psd_bw          = [psd_params[ii][2] for ii in range(len(psd_params))]

clean_db.clear()

#%% Simple hists to investigate data

input_var = resamp_zvals

fig = plt.figure(1)
plt.hist(input_var)

#%% Check: relationship between RD & PT symmetry

plt.scatter(mean_pt_sym, mean_rd_sym)
plt.scatter(median_pt_sym, median_rd_sym)
plt.xlabel('Peak-Through Sym')
plt.ylabel('Rise-Decay Sym')
plt.show()
# seems like small but consistent negative relationship

plt.hist(std_rd_sym)
plt.hist(std_pt_sym)
plt.xlabel('Standard deviation of symmetry (RT & PT)')
plt.show()
# PT sym has a smaller standard deviation



#%% HYPOTHESIS 1: THE SHAPE OF SIGNAL AFFECTS THE PAC-VALUE

# Two variables for using PAC values
# 1) Measure of correlation (Rho Value)
# 2) Statistical measure (Resampled Z Value)

# These values correlate with each other: 
plt.scatter(resamp_zvals, pac_rhos)
plt.xlabel('Resamp Z-value')
plt.ylabel('Rho Value')
plt.show()
# could this say something about the level of spurious pac in a signal?
# Rho value seems to be a slightly better feature because it does not have a cut-off

# for the shape of the signal we can use the symmetry measures
# We can used the RD of PT median or mean
# But we can also use the variation in the symmetry (STD)

plt.scatter(mean_pt_sym, median_pt_sym)
plt.xlabel('Mean PT sym')
plt.ylabel('Median PT sym')
plt.show()
# It does not really matter whether we use mean or median

plt.scatter(median_rd_sym, pac_rhos)
plt.scatter(median_pt_sym, pac_rhos)
plt.xlabel('Rho Value')
plt.ylabel('RD/PT sym')
plt.show()
# Especially PT seems to have a positive effect on the Rho value


plt.scatter(pac_rhos, std_rd_sym)
plt.xlabel('Rho Value')
plt.ylabel('RD Sym Standard deviation')
plt.show()
# Only channels with a lot variation in RD sym have a higher Rho value
# Which makes sense, but only the case in RD not PT

#%% Check: not really correlations between the different PSD parameters

plt.scatter(psd_bw,psd_cf)
plt.xlabel('BW')
plt.ylabel('CF')
plt.show()

plt.scatter(psd_amp,psd_cf)
plt.xlabel('AMP')
plt.ylabel('CF')
plt.show()

plt.scatter(psd_amp,psd_bw)
plt.xlabel('AMP')
plt.ylabel('BW')
plt.show()

#%% HYPOTHESIS 2: THE SPECTRAL PARAMETERS AFFECT THE PAC VALUE

plt.scatter(psd_cf, pac_rhos)
plt.xlabel('Rho Value')
plt.ylabel('CF')
plt.show()
# Higher Rho Values are only present in lower CFs. 


#%% HYPOTHESIS 3: SPECTRAL PARAMTERS & SIGNAL SYMMETRY

plt.scatter(psd_cf, median_pt_sym)
plt.scatter(psd_cf, median_rd_sym)
plt.xlabel('CF')
plt.ylabel('PT/RD symmetry')
plt.show()
# signals with higher central frequencies have nearly perfect symmetry

plt.scatter(psd_bw, median_pt_sym)
plt.scatter(psd_bw, median_rd_sym)
plt.xlabel('BW')
plt.ylabel('PT/RD symmetry')
plt.show()
# Same for BW, higher ones have more symmetry

#%% 

# BW is 2-8
# CF is 4-35



