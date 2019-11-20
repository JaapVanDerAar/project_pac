import os 
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn import linear_model
import pandas as pd

#%%  Load data 

# change dir 
os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab')

load_data = open('clean_db_nopac.pkl','rb')
clean_db_nopac = pickle.load(load_data)


subj_name   = clean_db_nopac['subj_name']
subj        = clean_db_nopac['subj']
ch          = clean_db_nopac['ch']
locs        = clean_db_nopac['locs']
dat_name    = clean_db_nopac['dat_name']
data        = clean_db_nopac['data']
pac_rhos_nopac    = clean_db_nopac['pac_rhos']
resamp_zvals_nopac = clean_db_nopac['resamp_zvals']
resamp_pvals_nopac = clean_db_nopac['resamp_pvals']
psd_params_nopac  = clean_db_nopac['psd_params']
backgr_params_nopac = clean_db_nopac['backgr_params']
rd_sym_nopac      = clean_db_nopac['rd_sym']
pt_sym_nopac      = clean_db_nopac['pt_sym']
bursts_nopac      = clean_db_nopac['bursts']
period_nopac      = clean_db_nopac['period']
volt_amp_nopac    = clean_db_nopac['volt_amp']
fs          = clean_db_nopac['fs']

mean_rd_sym_nopac     = [np.mean(rd_sym_nopac[ii]) for ii in range(len(rd_sym_nopac))]
median_rd_sym_nopac   = [np.median(rd_sym_nopac[ii]) for ii in range(len(rd_sym_nopac))]
std_rd_sym_nopac      = [np.std(rd_sym_nopac[ii]) for ii in range(len(rd_sym_nopac))]

mean_pt_sym_nopac     = [np.mean(pt_sym_nopac[ii]) for ii in range(len(pt_sym_nopac))]
median_pt_sym_nopac   = [np.median(pt_sym_nopac[ii]) for ii in range(len(pt_sym_nopac))]
std_pt_sym_nopac      = [np.std(pt_sym_nopac[ii]) for ii in range(len(pt_sym_nopac))]

median_volt_amp_nopac = [np.median(volt_amp_nopac[ii]) for ii in range(len(volt_amp_nopac))]
median_period_nopac   = [np.median(period_nopac[ii]) for ii in range(len(period_nopac))]

psd_cf_nopac          = [psd_params_nopac[ii][0] for ii in range(len(psd_params_nopac))]
psd_amp_nopac         = [psd_params_nopac[ii][1] for ii in range(len(psd_params_nopac))]
psd_bw_nopac          = [psd_params_nopac[ii][2] for ii in range(len(psd_params_nopac))]

backgr_offset_nopac = [backgr_params_nopac[ii][0] for ii in range(len(backgr_params_nopac))]
backgr_knee_nopac   = [backgr_params_nopac[ii][1] for ii in range(len(backgr_params_nopac))]
backgr_exp_nopac  = [backgr_params_nopac[ii][2] for ii in range(len(backgr_params_nopac))]

clean_db_nopac.clear()

#%% Put useful features into dataframe for easy plotting 

features_df_nopac = pd.DataFrame()
features_df_nopac['pac_rhos_nopac'] = pac_rhos_nopac 
features_df_nopac['resamp_zvals_nopac'] = resamp_zvals_nopac
features_df_nopac['median_rd_sym_nopac'] = median_rd_sym_nopac
features_df_nopac['median_pt_sym_nopac'] = median_pt_sym_nopac
features_df_nopac['psd_cf_nopac'] = psd_cf_nopac
features_df_nopac['psd_amp_nopac'] = psd_amp_nopac
features_df_nopac['psd_bw_nopac'] = psd_bw_nopac
features_df_nopac['backgr_exp_nopac'] = backgr_exp_nopac
features_df_nopac['backgr_offset_nopac'] = backgr_offset_nopac
features_df_nopac['backgr_knee_nopac'] = backgr_knee_nopac
features_df_nopac['median_volt_amp_nopac'] = median_volt_amp_nopac


#%% Check: relationship between RD & PT symmetry

plt.scatter(mean_pt_sym_nopac, mean_rd_sym_nopac, alpha=.5)
plt.scatter(median_pt_sym_nopac, median_rd_sym_nopac, alpha=.5)
plt.xlabel('Peak-Through Sym')
plt.ylabel('Rise-Decay Sym')
plt.title('Negative relationship between RD & PT')
plt.show()
# seems like small but consistent negative relationship

plt.hist(std_pt_sym_nopac,alpha=.5)
plt.hist(std_rd_sym_nopac,alpha=.5)
plt.xlabel('Standard deviation of symmetry (RT & PT)')
plt.title('smaller PT STDs')
plt.show()
# PT sym has a smaller standard deviation

#%% Reshape for sklearn

resamp_zvals_nopac = np.reshape(resamp_zvals_nopac, [len(resamp_zvals_nopac), 1])
pac_rhos_nopac = np.reshape(pac_rhos_nopac, [len(pac_rhos_nopac), 1])
psd_cf_nopac = np.reshape(psd_cf_nopac, [len(psd_cf_nopac), 1])
psd_bw_nopac = np.reshape(psd_bw_nopac, [len(psd_bw_nopac), 1])
median_pt_sym_nopac = np.reshape(median_pt_sym_nopac, [len(median_pt_sym_nopac), 1])
median_rd_sym_nopac  = np.reshape(median_rd_sym_nopac, [len(median_rd_sym_nopac), 1])
std_rd_sym_nopac  = np.reshape(std_rd_sym_nopac, [len(std_rd_sym_nopac), 1])

#%% HYPOTHESIS 1: THE SHAPE OF SIGNAL AFFECTS THE PAC-VALUE

# Two variables for using PAC values
# 1) Measure of correlation (Rho Value)
# 2) Statistical measure (Resampled Z Value)

# These values correlate with each other: 
plt.scatter(resamp_zvals_nopac, pac_rhos_nopac, alpha=.5)
plt.xlabel('Resamp Z-value')
plt.ylabel('Rho Value')
plt.title('Rho and resampled Z-value are correlated')
plt.show()
# could this say something about the level of spurious pac in a signal?
# Rho value seems to be a slightly better feature because it does not have a cut-off

# for the shape of the signal we can use the symmetry measures
# We can used the RD of PT median or mean
# But we can also use the variation in the symmetry (STD)

plt.scatter(mean_pt_sym_nopac, median_pt_sym_nopac, alpha=.5)
plt.xlabel('Mean PT sym')
plt.ylabel('Median PT sym')
plt.title('Median and Mean are strongly correlated')
plt.show()
# It does not really matter whether we use mean or median

reg1 = linear_model.LinearRegression()
reg1.fit(pac_rhos_nopac, median_pt_sym_nopac)
xs = np.arange(min(pac_rhos_nopac), max(pac_rhos_nopac),0.01)
ys = reg1.intercept_[0] + reg1.coef_[0][0] * xs
plt.plot(xs, ys, '--k', linewidth=4, label='Model', alpha=.5)

reg2 = linear_model.LinearRegression()
reg2.fit(pac_rhos_nopac, median_rd_sym_nopac)
xs = np.arange(min(pac_rhos_nopac), max(pac_rhos_nopac),0.01)
ys = reg2.intercept_[0] + reg2.coef_[0][0] * xs
plt.plot(xs, ys, '--k', linewidth=4, label='Model', alpha=.5)

plt.scatter(pac_rhos_nopac, median_pt_sym_nopac, alpha=.5)
plt.scatter(pac_rhos_nopac, median_rd_sym_nopac, alpha=.5)
plt.xlabel('Rho Value')
plt.ylabel('RD/PT sym')
plt.title('PT seems to have a positive effect on the Rho value')
plt.show()
# Especially PT seems to have a positive effect on the Rho value



reg = linear_model.LinearRegression()
reg.fit(pac_rhos_nopac, std_rd_sym_nopac)
xs = np.arange(min(pac_rhos_nopac), max(pac_rhos_nopac),0.01)
ys = reg.intercept_[0] + reg.coef_[0][0] * xs
plt.plot(xs, ys, '--k', linewidth=4, label='Model', alpha=.5)

plt.scatter(pac_rhos_nopac, std_rd_sym_nopac, alpha=.5)
plt.xlabel('Rho Value')
plt.ylabel('RD Sym Standard deviation')
plt.show()
# Only channels with a lot variation in RD sym have a higher Rho value
# Which makes sense, but only the case in RD not PT

#%% Check: not really correlations between the different PSD parameters

plt.scatter(psd_bw_nopac,psd_cf_nopac, alpha=.5)
plt.xlabel('BW')
plt.ylabel('CF')
plt.show()

plt.scatter(psd_amp_nopac,psd_cf_nopac, alpha=.5)
plt.xlabel('AMP')
plt.ylabel('CF')
plt.show()

plt.scatter(psd_amp_nopac,psd_bw_nopac, alpha=.5)
plt.xlabel('AMP')
plt.ylabel('BW')
plt.show()

#%% HYPOTHESIS 2: THE SPECTRAL PARAMETERS AFFECT THE PAC VALUE

reg1 = linear_model.LinearRegression()
reg1.fit(psd_cf_nopac, pac_rhos_nopac)
xs = np.arange(min(psd_cf_nopac), max(psd_cf_nopac))
ys = reg1.intercept_[0] + reg1.coef_[0][0] * xs
plt.plot(xs, ys, '--k', linewidth=4, label='Model', alpha=.5)

plt.scatter(psd_cf_nopac, pac_rhos_nopac, alpha=.5)
plt.xlabel('CF')
plt.ylabel('Rho value')
plt.title('Higher Rho Values are only present in lower CFs')
plt.show()
# Higher Rho Values are only present in lower CFs. 


#%% HYPOTHESIS 3: SPECTRAL PARAMTERS & SIGNAL SYMMETRY

reg1 = linear_model.LinearRegression()
reg1.fit(psd_cf_nopac, median_pt_sym_nopac)
xs = np.arange(min(psd_cf_nopac), max(psd_cf_nopac))
ys = reg1.intercept_[0] + reg1.coef_[0][0] * xs
plt.plot(xs, ys, '--k', linewidth=4, label='Model', alpha=.5)


reg2 = linear_model.LinearRegression()
reg2.fit(psd_cf_nopac, median_rd_sym_nopac)
xs = np.arange(min(psd_cf_nopac), max(psd_cf_nopac))
ys = reg2.intercept_[0] + reg2.coef_[0][0] * xs
plt.plot(xs, ys, '--k', linewidth=4, label='Model', alpha=.5)

plt.scatter(psd_cf_nopac, median_pt_sym_nopac , alpha=.5)
plt.scatter(psd_cf_nopac, median_rd_sym_nopac, alpha=.5)
plt.xlabel('CF')
plt.ylabel('PT/RD symmetry')
plt.title('signals with higher CF have nearly perfect symmetry')
plt.show()
# signals with higher central frequencies have nearly perfect symmetry


reg1 = linear_model.LinearRegression()
reg1.fit(psd_bw_nopac, median_pt_sym_nopac)
xs = np.arange(min(psd_bw_nopac), max(psd_bw_nopac))
ys = reg1.intercept_[0] + reg1.coef_[0][0] * xs
plt.plot(xs, ys, '--k', linewidth=4, label='Model', alpha=.5)


reg2 = linear_model.LinearRegression()
reg2.fit(psd_bw_nopac, median_rd_sym_nopac)
xs = np.arange(min(psd_bw_nopac), max(psd_bw_nopac))
ys = reg2.intercept_[0] + reg2.coef_[0][0] * xs
plt.plot(xs, ys, '--k', linewidth=4, label='Model', alpha=.5)

plt.scatter(psd_bw_nopac, median_pt_sym_nopac, alpha=.5)
plt.scatter(psd_bw_nopac, median_rd_sym_nopac, alpha=.5)

plt.xlabel('BW')
plt.ylabel('PT/RD symmetry')
plt.title('Same for BW, higher ones have more symmetry')
plt.show()
# Same for BW, higher ones have more symmetry

#%% Clustering Symmetry with Rho data

plt.figure(figsize=(15,10))

# Blue Color: low PT and low Rho
plt.scatter(features_df_nopac["pac_rhos_nopac"][~(features_df_nopac["pac_rhos_nopac"] >.10) |  ~(features_df_nopac["median_pt_sym_nopac"] >.55)],
            features_df_nopac["median_pt_sym_nopac"][~(features_df_nopac["pac_rhos_nopac"] >.10) | ~(features_df_nopac["median_pt_sym_nopac"] >.55)], 
            c='blue', label='low PT low Rho')

# Red Color: low PT and high Rho 
plt.scatter(features_df_nopac["pac_rhos_nopac"][(features_df_nopac["pac_rhos_nopac"] >.10)],
            features_df_nopac["median_pt_sym_nopac"][(features_df_nopac["pac_rhos_nopac"] >.10)],
            c='red', label='low PT high Rho')

# Yellow Color: High PT low Rho
plt.scatter(features_df_nopac["pac_rhos_nopac"][(features_df_nopac["median_pt_sym_nopac"] >.55)],
            features_df_nopac["median_pt_sym_nopac"][(features_df_nopac["median_pt_sym_nopac"] >.55)], 
            c='yellow', label='High PT low Rho')

# Green Color: High PT high Rho
plt.scatter(features_df_nopac["pac_rhos_nopac"][(features_df_nopac["pac_rhos_nopac"] >.10) & (features_df_nopac["median_pt_sym_nopac"] >.55)],
            features_df_nopac["median_pt_sym_nopac"][(features_df_nopac["pac_rhos_nopac"] >.10) & (features_df_nopac["median_pt_sym_nopac"] >.55)], 
            c='green', label='High PT high Rho')


plt.xlabel('Rho Value')
plt.ylabel('PT sym')
plt.title('Rho values have a high PT symmetry. If they have a low PT they have a low Rho')
plt.legend(scatterpoints=1, loc='upper left');
plt.show()



#See where those datapoints are if we use them on the RD scatter

plt.figure(figsize=(15,10))

# Blue Color: low PT and low Rho
plt.scatter(features_df_nopac["pac_rhos_nopac"][~(features_df_nopac["pac_rhos_nopac"] >.10) |  ~(features_df_nopac["median_pt_sym_nopac"] >.55)],
            features_df_nopac["median_rd_sym_nopac"][~(features_df_nopac["pac_rhos_nopac"] >.10) | ~(features_df_nopac["median_pt_sym_nopac"] >.55)], 
            c='blue', label='low PT low Rho')

# Red Color: low PT and high Rho 
plt.scatter(features_df_nopac["pac_rhos_nopac"][(features_df_nopac["pac_rhos_nopac"] >.10)],
            features_df_nopac["median_rd_sym_nopac"][(features_df_nopac["pac_rhos_nopac"] >.10)],
            c='red', label='low PT high Rho')

# Yellow Color: High PT low Rho
plt.scatter(features_df_nopac["pac_rhos_nopac"][(features_df_nopac["median_pt_sym_nopac"] >.55)],
            features_df_nopac["median_rd_sym_nopac"][(features_df_nopac["median_pt_sym_nopac"] >.55)], 
            c='yellow', label='High PT low Rho')

# Green Color: High PT high Rho
plt.scatter(features_df_nopac["pac_rhos_nopac"][(features_df_nopac["pac_rhos_nopac"] >.10) & (features_df_nopac["median_pt_sym_nopac"] >.55)],
            features_df_nopac["median_rd_sym_nopac"][(features_df_nopac["pac_rhos_nopac"] >.10) & (features_df_nopac["median_pt_sym_nopac"] >.55)], 
            c='green', label='High PT high Rho')


plt.xlabel('Rho Value')
plt.ylabel('RD sym')
plt.title('Using same datapoint colors: Some of this structure also found in RD Sym, but less consistent')
plt.legend(scatterpoints=1, loc='upper left');
plt.show()


#See where those datapoints are when we use CF scatter

plt.figure(figsize=(15,10))

# Blue Color: low PT and low Rho
plt.scatter(features_df_nopac["pac_rhos_nopac"][~(features_df_nopac["pac_rhos_nopac"] >.10) |  ~(features_df_nopac["median_pt_sym_nopac"] >.55)],
            features_df_nopac["psd_cf_nopac"][~(features_df_nopac["pac_rhos_nopac"] >.10) | ~(features_df_nopac["median_pt_sym_nopac"] >.55)], 
            c='blue', label='low PT low Rho')

# Red Color: low PT and high Rho 
plt.scatter(features_df_nopac["pac_rhos_nopac"][(features_df_nopac["pac_rhos_nopac"] >.10)],
            features_df_nopac["psd_cf_nopac"][(features_df_nopac["pac_rhos_nopac"] >.10)],
            c='red', label='low PT high Rho')

# Yellow Color: High PT low Rho
plt.scatter(features_df_nopac["pac_rhos_nopac"][(features_df_nopac["median_pt_sym_nopac"] >.55)],
            features_df_nopac["psd_cf_nopac"][(features_df_nopac["median_pt_sym_nopac"] >.55)], 
            c='yellow', label='High PT low Rho')

# Green Color: High PT high Rho
plt.scatter(features_df_nopac["pac_rhos_nopac"][(features_df_nopac["pac_rhos_nopac"] >.10) & (features_df_nopac["median_pt_sym_nopac"] >.55)],
            features_df_nopac["psd_cf_nopac"][(features_df_nopac["pac_rhos_nopac"] >.10) & (features_df_nopac["median_pt_sym_nopac"] >.55)], 
            c='green', label='High PT high Rho')


plt.xlabel('Rho Value')
plt.ylabel('CF')
plt.title('High PT is only found in lower CFs. Rho Values > .08 are only found in lower CFs')
plt.legend(scatterpoints=1, loc='upper left');
plt.show()



#%% Give Rho's and resampled Z-values specific colors and see how their values
### Are related to other parameters


# give values 
rho_low = .17
rho_hi = .17
zval_low = 10.5
zval_hi = 10.5


plt.figure(figsize=(15,10))

# Blue Color: low Zval and low Rho
plt.scatter(features_df_nopac["pac_rhos_nopac"][~(features_df_nopac["pac_rhos_nopac"] > rho_low) |  ~(features_df_nopac["resamp_zvals_nopac"] > zval_low)],
            features_df_nopac["resamp_zvals_nopac"][~(features_df_nopac["pac_rhos_nopac"] > rho_low) | ~(features_df_nopac["resamp_zvals_nopac"] > zval_low)], 
            c='blue', label='low Zval low Rho')

# Red Color: low PT and high Rho 
plt.scatter(features_df_nopac["pac_rhos_nopac"][(features_df_nopac["pac_rhos_nopac"] > rho_low)],
            features_df_nopac["resamp_zvals_nopac"][(features_df_nopac["pac_rhos_nopac"] > rho_low)],
            c='red', label='low Zval high Rho')

# Yellow Color: High PT low Rho
plt.scatter(features_df_nopac["pac_rhos_nopac"][(features_df_nopac["resamp_zvals_nopac"] > zval_low)],
            features_df_nopac["resamp_zvals_nopac"][(features_df_nopac["resamp_zvals_nopac"] > zval_low)], 
            c='yellow', label='High Zval low Rho')

# Green Color: High PT high Rho
plt.scatter(features_df_nopac["pac_rhos_nopac"][(features_df_nopac["pac_rhos_nopac"] > rho_hi) & (features_df_nopac["resamp_zvals_nopac"] > zval_hi)],
            features_df_nopac["resamp_zvals_nopac"][(features_df_nopac["pac_rhos_nopac"] > rho_hi) & (features_df_nopac["resamp_zvals_nopac"] > zval_hi)], 
            c='green', label='High Zval high Rho')

plt.xlabel('Rho Value')
plt.ylabel('Resamp Z-value')
plt.title('Categorizing the outcome measures, although they are strongly related')
plt.legend(scatterpoints=1, loc='upper left');
plt.show()

#%% The outcome measures/statistics are somewhat related to the Symmetry measures

plt.figure(figsize=(15,10))

# Blue Color: low Zval and low Rho
plt.scatter(features_df_nopac["median_pt_sym_nopac"][~(features_df_nopac["pac_rhos_nopac"] > rho_low) |  ~(features_df_nopac["resamp_zvals_nopac"] > zval_low)],
            features_df_nopac["median_rd_sym_nopac"][~(features_df_nopac["pac_rhos_nopac"] > rho_low) | ~(features_df_nopac["resamp_zvals_nopac"] > zval_low)], 
            c='blue', label='low Zval low Rho')

# Red Color: low PT and high Rho 
plt.scatter(features_df_nopac["median_pt_sym_nopac"][(features_df_nopac["pac_rhos_nopac"] > rho_low)],
            features_df_nopac["median_rd_sym_nopac"][(features_df_nopac["pac_rhos_nopac"] > rho_low)],
            c='red', label='low Zval high Rho')

# Yellow Color: High PT low Rho
plt.scatter(features_df_nopac["median_pt_sym_nopac"][(features_df_nopac["resamp_zvals_nopac"] > zval_low)],
            features_df_nopac["median_rd_sym_nopac"][(features_df_nopac["resamp_zvals_nopac"] > zval_low)], 
            c='yellow', label='High Zval low Rho')

# Green Color: High PT high Rho
plt.scatter(features_df_nopac["median_pt_sym_nopac"][(features_df_nopac["pac_rhos_nopac"] > rho_hi) & (features_df_nopac["resamp_zvals_nopac"] > zval_hi)],
            features_df_nopac["median_rd_sym_nopac"][(features_df_nopac["pac_rhos_nopac"] > rho_hi) & (features_df_nopac["resamp_zvals_nopac"] > zval_hi)], 
            c='green', label='High Zval high Rho')

plt.xlabel('PT Symmetry')
plt.ylabel('RD Symmetry')
plt.title('The outcome measures/statistics are somewhat related to the Symmetry measures')
plt.legend(scatterpoints=1, loc='upper left');
plt.show()

#%% High outcome measures have a high PT sym and a low CF

plt.figure(figsize=(15,10))

# Blue Color: low Zval and low Rho
plt.scatter(features_df_nopac["median_pt_sym_nopac"][~(features_df_nopac["pac_rhos_nopac"] > rho_low) |  ~(features_df_nopac["resamp_zvals_nopac"] > zval_low)],
            features_df_nopac["psd_cf_nopac"][~(features_df_nopac["pac_rhos_nopac"] > rho_low) | ~(features_df_nopac["resamp_zvals_nopac"] > zval_low)], 
            c='blue', label='low Zval low Rho')

# Red Color: low PT and high Rho 
plt.scatter(features_df_nopac["median_pt_sym_nopac"][(features_df_nopac["pac_rhos_nopac"] > rho_low)],
            features_df_nopac["psd_cf_nopac"][(features_df_nopac["pac_rhos_nopac"] > rho_low)],
            c='red', label='low Zval high Rho')

# Yellow Color: High PT low Rho
plt.scatter(features_df_nopac["median_pt_sym_nopac"][(features_df_nopac["resamp_zvals_nopac"] > zval_low)],
            features_df_nopac["psd_cf_nopac"][(features_df_nopac["resamp_zvals_nopac"] > zval_low)], 
            c='yellow', label='High Zval low Rho')

# Green Color: High PT high Rho
plt.scatter(features_df_nopac["median_pt_sym_nopac"][(features_df_nopac["pac_rhos_nopac"] > rho_hi) & (features_df_nopac["resamp_zvals_nopac"] > zval_hi)],
            features_df_nopac["psd_cf_nopac"][(features_df_nopac["pac_rhos_nopac"] > rho_hi) & (features_df_nopac["resamp_zvals_nopac"] > zval_hi)], 
            c='green', label='High Zval high Rho')

plt.xlabel('PT Symmetry')
plt.ylabel('CF')
plt.title('High outcome measures have a high PT sym and a low CF')
plt.legend(scatterpoints=1, loc='upper left');
plt.show()

#%% Higher outcome measures have less variation in symmetry (especially in RD)

plt.figure(figsize=(15,10))

# Blue Color: low Zval and low Rho
plt.scatter(features_df_nopac["std_rd_sym_nopac"][~(features_df_nopac["pac_rhos_nopac"] > rho_low) |  ~(features_df_nopac["resamp_zvals_nopac"] > zval_low)],
            features_df_nopac["pac_rhos_nopac"][~(features_df_nopac["pac_rhos_nopac"] > rho_low) | ~(features_df_nopac["resamp_zvals_nopac"] > zval_low)], 
            c='blue', label='low Zval low Rho')

# Red Color: low PT and high Rho 
plt.scatter(features_df_nopac["std_rd_sym_nopac"][(features_df_nopac["pac_rhos_nopac"] > rho_low)],
            features_df_nopac["pac_rhos_nopac"][(features_df_nopac["pac_rhos_nopac"] > rho_low)],
            c='red', label='low Zval high Rho')

# Yellow Color: High PT low Rho
plt.scatter(features_df_nopac["std_rd_sym_nopac"][(features_df_nopac["resamp_zvals_nopac"] > zval_low)],
            features_df_nopac["pac_rhos_nopac"][(features_df_nopac["resamp_zvals_nopac"] > zval_low)], 
            c='yellow', label='High Zval low Rho')

# Green Color: High PT high Rho
plt.scatter(features_df_nopac["std_rd_sym_nopac"][(features_df_nopac["pac_rhos_nopac"] > rho_hi) & (features_df_nopac["resamp_zvals_nopac"] > zval_hi)],
            features_df_nopac["pac_rhos_nopac"][(features_df_nopac["pac_rhos_nopac"] > rho_hi) & (features_df_nopac["resamp_zvals_nopac"] > zval_hi)], 
            c='green', label='High Zval high Rho')

plt.xlabel('RD STD')
plt.ylabel('Rho Value')
plt.title('High outcome measures have less variation in the RD symmetry')
plt.legend(scatterpoints=1, loc='upper left');
plt.show()

#%% Higher outcome measures have less variation in symmetry (especially in RD)

plt.figure(figsize=(15,10))

# Blue Color: low Zval and low Rho
plt.scatter(features_df_nopac["std_rd_sym_nopac"][~(features_df_nopac["pac_rhos_nopac"] > rho_low) |  ~(features_df_nopac["resamp_zvals_nopac"] > zval_low)],
            features_df_nopac["median_pt_sym_nopac"][~(features_df_nopac["pac_rhos_nopac"] > rho_low) | ~(features_df_nopac["resamp_zvals_nopac"] > zval_low)], 
            c='blue', label='low Zval low Rho')

# Red Color: low PT and high Rho 
plt.scatter(features_df_nopac["std_rd_sym_nopac"][(features_df_nopac["pac_rhos_nopac"] > rho_low)],
            features_df_nopac["median_pt_sym_nopac"][(features_df_nopac["pac_rhos_nopac"] > rho_low)],
            c='red', label='low Zval high Rho')

# Yellow Color: High PT low Rho
plt.scatter(features_df_nopac["std_rd_sym_nopac"][(features_df_nopac["resamp_zvals_nopac"] > zval_low)],
            features_df_nopac["median_pt_sym_nopac"][(features_df_nopac["resamp_zvals_nopac"] > zval_low)], 
            c='yellow', label='High Zval low Rho')

# Green Color: High PT high Rho
plt.scatter(features_df_nopac["std_rd_sym_nopac"][(features_df_nopac["pac_rhos_nopac"] > rho_hi) & (features_df_nopac["resamp_zvals_nopac"] > zval_hi)],
            features_df_nopac["median_pt_sym_nopac"][(features_df_nopac["pac_rhos_nopac"] > rho_hi) & (features_df_nopac["resamp_zvals_nopac"] > zval_hi)], 
            c='green', label='High Zval high Rho')

plt.xlabel('RD STD')
plt.ylabel('median_pt_sym_nopac')
plt.title('High outcome measures have less variation in the RD symmetry')
plt.legend(scatterpoints=1, loc='upper left');
plt.show()

#%% NEXT STEPS: 

# 1) get all features
# 2) look at colloniarity (correlations) and select features
# 3) do PCA
# 4) unsupervised clustering



#%% Inspect features

pd.plotting.scatter_matrix(features_df_nopac, figsize=(40,40))


#%%


f = plt.figure(figsize=(19, 15))
plt.matshow(features_df_nopac.corr(), fignum=f.number)
plt.xticks(range(features_df_nopac.shape[1]), features_df_nopac.columns, fontsize=14, rotation=45)
plt.yticks(range(features_df_nopac.shape[1]), features_df_nopac.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16);

#%% adjust features in features_df_nopac

# drop backgr_knee_nopac, because no correlations with other data, 
# and has homogeneous data

features_df_nopac = features_df_nopac.drop(columns='backgr_knee_nopac')

# Log transfer psd_bw_nopac
features_df_nopac['psd_bw_nopac_log'] = np.log10(features_df_nopac['psd_bw_nopac'])
# drop old psd_bw_nopac for clean df
features_df_nopac = features_df_nopac.drop(columns='psd_bw_nopac')

# and create 1 feature out of pac_rhos_nopac + resamp_zvals_nopac
# and create 1 feature out of backgr_exp_nopac and offset


#%% Scale data & PCA example

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import mglearn


#%% Set data to right structure for ML with sklearn

# first, logtransform bandwidth
psd_bw_nopac_log10 = np.log10(psd_bw_nopac)

# change to right shape for sklearn
pac_rhos_nopac = np.reshape(pac_rhos_nopac, [len(pac_rhos_nopac), 1])
resamp_zvals_nopac = np.reshape(resamp_zvals_nopac, [len(resamp_zvals_nopac), 1])
median_rd_sym_nopac  = np.reshape(median_rd_sym_nopac, [len(median_rd_sym_nopac), 1])
median_pt_sym_nopac = np.reshape(median_pt_sym_nopac, [len(median_pt_sym_nopac), 1])
psd_cf_nopac = np.reshape(psd_cf_nopac, [len(psd_cf_nopac), 1])
psd_amp_nopac = np.reshape(psd_amp_nopac, [len(psd_amp_nopac), 1])
backgr_exp_nopac = np.reshape(backgr_exp_nopac, [len(backgr_exp_nopac), 1])
backgr_offset_nopac = np.reshape(backgr_offset_nopac, [len(backgr_offset_nopac), 1])
median_volt_amp_nopac = np.reshape(median_volt_amp_nopac, [len(median_volt_amp_nopac), 1])
psd_bw_nopac_log10 = np.reshape(psd_bw_nopac_log10, [len(psd_bw_nopac_log10), 1])

# and create 1 feature out of pac_rhos_nopac + resamp_zvals_nopac
scaler = StandardScaler()
pac_values_nopac = scaler.fit_transform(pac_rhos_nopac) + scaler.fit_transform(resamp_zvals_nopac)

# and create 1 feature out of backgr_exp_nopac and offset
scaler = StandardScaler()
aperiodic_param_nopac = scaler.fit_transform(backgr_exp_nopac) + scaler.fit_transform(backgr_offset_nopac)

# change to right shape for sklearn 
aperiodic_param_nopac = np.reshape(aperiodic_param_nopac, [len(aperiodic_param_nopac), 1])
pac_values_nopac = np.reshape(pac_values_nopac, [len(pac_values_nopac), 1])

pac_features_nopac = np.hstack((pac_values_nopac, median_rd_sym_nopac, median_pt_sym_nopac, median_volt_amp_nopac, \
                           aperiodic_param_nopac, psd_cf_nopac, psd_amp_nopac, psd_bw_nopac_log10))

# scale data
scaler = StandardScaler()
scaler.fit(pac_features_nopac)
X_scaled_nopac = scaler.transform(pac_features_nopac)

#%% PAC Scale data & PCA 


feature_list_nopac = ['pac_values_nopac', 'median_rd_sym_nopac', 'median_pt_sym_nopac', 'median_volt_amp_nopac', \
                'aperiodic_param_nopac', 'psd_cf_nopac', 'psd_amp_nopac', 'psd_bw_nopac_log10']
# scale data
scaler = StandardScaler()
scaler.fit(pac_features_nopac)
X_scaled_nopac = scaler.transform(pac_features_nopac)

# PCA
pca = PCA(n_components=2)
pca.fit(X_scaled_nopac)
X_pca = pca.transform(X_scaled_nopac)

# plot data based on components
plt.figure(figsize=(8, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.title('Plot data based on components')
plt.xlabel("First principal component")
plt.ylabel("Second principal component")

# insight in components
plt.matshow(pca.components_, cmap='viridis')
plt.yticks([0, 1], ["First component", "Second component"])
plt.colorbar()
plt.xticks(range(len(feature_list_nopac)),feature_list_nopac, rotation=60, ha='left')
plt.xlabel("Feature")
plt.ylabel("Principal components")


#%% K-means

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
kmeans.fit(X_scaled_nopac)

# visualize on PCA  
mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], kmeans.labels_, markers='o')
mglearn.discrete_scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], [0, 1], markers='^', markeredgewidth=3)
plt.title('K-Means: Plot clusters visualized on PCA\'s including center')
plt.xlabel("First principal component")
plt.ylabel("Second principal component")

features_df_nopac['Clusters'] = kmeans.labels_
cluster_label = kmeans.labels_

#%% Agglomerative clustering

from sklearn.cluster import AgglomerativeClustering
agg = AgglomerativeClustering(n_clusters=2)
assignment = agg.fit_predict(X_scaled_nopac)
mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], assignment)
plt.title('AggloClustering: Plot clusters visualized on PCA\'s') 
plt.xlabel("Component 1")
plt.ylabel("Component 2")


#%% Hierarchical cluster / Dendogram

from scipy.cluster.hierarchy import dendrogram, ward

linkage_array = ward(X_scaled_nopac)

# Now we plot the dendrogram for the linkage_array containing the distances
# between clusters
dendrogram(linkage_array)
# Mark the cuts in the tree that signify two or three clusters
ax = plt.gca()
bounds = ax.get_xbound()
plt.title('Might consider 3 clusters')
plt.ylabel("Cluster distance")

#%% DBSCAN

from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.1,min_samples=2)
clusters = dbscan.fit_predict(X_scaled_nopac)
# plot the cluster assignments
plt.scatter(X_scaled_nopac[:, 0], X_pca[:, 1])#, c=clusters, cmap=mglearn.cm2, s=60)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")


#%% Linear regression predicting pac_values_nopac in each cluster

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
kmeans.fit(X_scaled_nopac)

# or without pac_values_nopac included
# kmeans.fit(X_scaled_nopac[:,1:])

# visualize on PCA  
mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], kmeans.labels_, markers='o')
mglearn.discrete_scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], [0, 1], markers='^', markeredgewidth=3)
plt.title('K-Means: Plot clusters visualized on PCA\'s including center')
plt.xlabel("First principal component")
plt.ylabel("Second principal component")

features_df_nopac['Clusters'] = kmeans.labels_
cluster_label = kmeans.labels_


# which channels/data are in which cluster
features_cluster0 = np.zeros([(cluster_label==0).sum(),len(X_scaled_nopac[0])])
features_cluster1 = np.zeros([(cluster_label==1).sum(),len(X_scaled_nopac[0])])

for ii in range(len(X_scaled_nopac[0])):
    features_cluster0[:,ii] = [X_scaled_nopac[jj,ii] for jj in range(len(X_scaled_nopac)) if cluster_label[jj] == 0]
    features_cluster1[:,ii] = [X_scaled_nopac[jj,ii] for jj in range(len(X_scaled_nopac)) if cluster_label[jj] == 1]
    
# get X and Y_hat (splits features into pac_value and other features)
X_cluster0 = features_cluster0[:,1:]
X_cluster1 = features_cluster1[:,1:]

Y_cluster0 = features_cluster0[:,0]
Y_cluster1 = features_cluster1[:,0]



# regression fit
reg_cluster0 = linear_model.LinearRegression().fit(X_cluster0, Y_cluster0)
#print(reg_cluster0.coef_)
reg_cluster1 = linear_model.LinearRegression().fit(X_cluster1, Y_cluster1)
#print(reg_cluster1.coef_)


# create plot
#fig, ax = plt.subplots()
plt.figure(figsize=(20,8))
index = np.arange(7)
bar_width = 0.25
opacity = 0.8

rects1 = plt.bar(index, reg_cluster0.coef_, bar_width,
color='b',
label='Cluster 0')

rects2 = plt.bar(index + bar_width, reg_cluster1.coef_, bar_width,
color='r',
label='Cluster 1')

beta_difference = abs(reg_cluster1.coef_ - reg_cluster0.coef_)

rects3 = plt.bar(index + (bar_width * 2), beta_difference, bar_width,
color='g',
label='Difference')

plt.xlabel('Feature')
plt.ylabel('Beta Value',fontsize=16)
plt.title('Linear regression: features on PAC-value',fontsize=16)
plt.xticks(index + bar_width, feature_list_nopac[1:8],fontsize=16)
plt.legend(fontsize=16)

#plt.tight_layout()
plt.show()




#%%% Plot difference in features between NoPacs and PACs in histograms

# manually set which columns you want to plot
plot_list = [0,1,2,3,4,5,6,8,9]



plt.figure(figsize=(20,20))
for ii in range(len(plot_list)): 
        
    jj = plot_list[ii]
    # subplots 2x5
    plt.subplot(3,3,ii+1)
    #xticks([]), yticks([])
    plt.title(features_df_nopac.columns[jj])
    plt.hist(features_df_nopac.iloc[:,jj], alpha=.4)
    plt.axvline(np.median(features_df_nopac.iloc[:,jj]), color='b', linestyle='dashed', linewidth=1)
    
    plt.hist(features_df.iloc[:,jj], alpha=.4)
    plt.axvline(np.median(features_df.iloc[:,jj]), color='r', linestyle='dashed', linewidth=1)
    








#%% ML - Supervised


# Prediction: Rho value
# Features: PT, RD, STD's, CF, BW

# pac_rhos_nopac = np.reshape(pac_rhos_nopac, [len(pac_rhos_nopac), 1])
psd_cf_nopac = np.reshape(psd_cf_nopac, [len(psd_cf_nopac), 1])
psd_bw_nopac = np.reshape(psd_bw_nopac, [len(psd_bw_nopac), 1])
median_pt_sym_nopac = np.reshape(median_pt_sym_nopac, [len(median_pt_sym_nopac), 1])
median_rd_sym_nopac  = np.reshape(median_rd_sym_nopac, [len(median_rd_sym_nopac), 1])
std_rd_sym_nopac  = np.reshape(std_rd_sym_nopac, [len(std_rd_sym_nopac), 1])
std_pt_sym_nopac  = np.reshape(std_pt_sym_nopac, [len(std_rd_sym_nopac), 1])

pac_rhos_nopac_binary = []
for ii in range(len(pac_rhos_nopac)):
    if pac_rhos_nopac[ii] > 0.1:
        pac_rhos_nopac_binary_0 = 1
    else: 
        pac_rhos_nopac_binary_0 = 0
        
    pac_rhos_nopac_binary.append(pac_rhos_nopac_binary_0)
        
pac_rhos_nopac_binary = np.reshape(pac_rhos_nopac_binary, [len(pac_rhos_nopac_binary)])

x_hat = np.hstack((median_pt_sym_nopac, median_rd_sym_nopac, std_pt_sym_nopac, std_rd_sym_nopac, psd_cf_nopac, psd_bw_nopac))
y_hat = pac_rhos_nopac_binary

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_hat,y_hat)

#%% ML - Unsupervised
# first scale features

