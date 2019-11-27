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
backgr_params = clean_db['backgr_params']
rd_sym      = clean_db['rd_sym']
pt_sym      = clean_db['pt_sym']
bursts      = clean_db['bursts']
period      = clean_db['period']
volt_amp    = clean_db['volt_amp']
fs          = clean_db['fs']

mean_rd_sym     = [np.mean(rd_sym[ii]) for ii in range(len(rd_sym))]
median_rd_sym   = [np.median(rd_sym[ii]) for ii in range(len(rd_sym))]
std_rd_sym      = [np.std(rd_sym[ii]) for ii in range(len(rd_sym))]

mean_pt_sym     = [np.mean(pt_sym[ii]) for ii in range(len(pt_sym))]
median_pt_sym   = [np.median(pt_sym[ii]) for ii in range(len(pt_sym))]
std_pt_sym      = [np.std(pt_sym[ii]) for ii in range(len(pt_sym))]

median_volt_amp = [np.median(volt_amp[ii]) for ii in range(len(volt_amp))]
median_period   = [np.median(period[ii]) for ii in range(len(period))]

psd_cf          = [psd_params[ii][0] for ii in range(len(psd_params))]
psd_amp         = [psd_params[ii][1] for ii in range(len(psd_params))]
psd_bw          = [psd_params[ii][2] for ii in range(len(psd_params))]

backgr_offset = [backgr_params[ii][0] for ii in range(len(backgr_params))]
backgr_knee   = [backgr_params[ii][1] for ii in range(len(backgr_params))]
backgr_exp  = [backgr_params[ii][2] for ii in range(len(backgr_params))]

clean_db.clear()

#%% Put useful features into dataframe for easy plotting 

features_df = pd.DataFrame()
features_df['pac_rhos'] = pac_rhos 
features_df['resamp_zvals'] = resamp_zvals
features_df['median_rd_sym'] = median_rd_sym
features_df['median_pt_sym'] = median_pt_sym
features_df['psd_cf'] = psd_cf
features_df['psd_amp'] = psd_amp
features_df['psd_bw'] = psd_bw
features_df['backgr_exp'] = backgr_exp
features_df['backgr_offset'] = backgr_offset
features_df['backgr_knee'] = backgr_knee
features_df['median_volt_amp'] = median_volt_amp


#%% Check: relationship between RD & PT symmetry

plt.scatter(mean_pt_sym, mean_rd_sym, alpha=.5)
plt.scatter(median_pt_sym, median_rd_sym, alpha=.5)
plt.xlabel('Peak-Through Sym')
plt.ylabel('Rise-Decay Sym')
plt.title('Negative relationship between RD & PT')
plt.show()
# seems like small but consistent negative relationship

plt.hist(std_pt_sym,alpha=.5)
plt.hist(std_rd_sym,alpha=.5)
plt.xlabel('Standard deviation of symmetry (RT & PT)')
plt.title('smaller PT STDs')
plt.show()
# PT sym has a smaller standard deviation

#%% Reshape for sklearn

resamp_zvals = np.reshape(resamp_zvals, [len(resamp_zvals), 1])
pac_rhos = np.reshape(pac_rhos, [len(pac_rhos), 1])
psd_cf = np.reshape(psd_cf, [len(psd_cf), 1])
psd_bw = np.reshape(psd_bw, [len(psd_bw), 1])
median_pt_sym = np.reshape(median_pt_sym, [len(median_pt_sym), 1])
median_rd_sym  = np.reshape(median_rd_sym, [len(median_rd_sym), 1])
std_rd_sym  = np.reshape(std_rd_sym, [len(std_rd_sym), 1])

#%% HYPOTHESIS 1: THE SHAPE OF SIGNAL AFFECTS THE PAC-VALUE

# Two variables for using PAC values
# 1) Measure of correlation (Rho Value)
# 2) Statistical measure (Resampled Z Value)

# These values correlate with each other: 
plt.scatter(resamp_zvals, pac_rhos, alpha=.5)
plt.xlabel('Resamp Z-value')
plt.ylabel('Rho Value')
plt.title('Rho and resampled Z-value are correlated')
plt.show()
# could this say something about the level of spurious pac in a signal?
# Rho value seems to be a slightly better feature because it does not have a cut-off

# for the shape of the signal we can use the symmetry measures
# We can used the RD of PT median or mean
# But we can also use the variation in the symmetry (STD)

plt.scatter(mean_pt_sym, median_pt_sym, alpha=.5)
plt.xlabel('Mean PT sym')
plt.ylabel('Median PT sym')
plt.title('Median and Mean are strongly correlated')
plt.show()
# It does not really matter whether we use mean or median

reg1 = linear_model.LinearRegression()
reg1.fit(pac_rhos, median_pt_sym)
xs = np.arange(min(pac_rhos), max(pac_rhos),0.01)
ys = reg1.intercept_[0] + reg1.coef_[0][0] * xs
plt.plot(xs, ys, '--k', linewidth=4, label='Model', alpha=.5)

reg2 = linear_model.LinearRegression()
reg2.fit(pac_rhos, median_rd_sym)
xs = np.arange(min(pac_rhos), max(pac_rhos),0.01)
ys = reg2.intercept_[0] + reg2.coef_[0][0] * xs
plt.plot(xs, ys, '--k', linewidth=4, label='Model', alpha=.5)

plt.scatter(pac_rhos, median_pt_sym, alpha=.5)
plt.scatter(pac_rhos, median_rd_sym, alpha=.5)
plt.xlabel('Rho Value')
plt.ylabel('RD/PT sym')
plt.title('PT seems to have a positive effect on the Rho value')
plt.show()
# Especially PT seems to have a positive effect on the Rho value



reg = linear_model.LinearRegression()
reg.fit(pac_rhos, std_rd_sym)
xs = np.arange(min(pac_rhos), max(pac_rhos),0.01)
ys = reg.intercept_[0] + reg.coef_[0][0] * xs
plt.plot(xs, ys, '--k', linewidth=4, label='Model', alpha=.5)

plt.scatter(pac_rhos, std_rd_sym, alpha=.5)
plt.xlabel('Rho Value')
plt.ylabel('RD Sym Standard deviation')
plt.show()
# Only channels with a lot variation in RD sym have a higher Rho value
# Which makes sense, but only the case in RD not PT

#%% Check: not really correlations between the different PSD parameters

plt.scatter(psd_bw,psd_cf, alpha=.5)
plt.xlabel('BW')
plt.ylabel('CF')
plt.show()

plt.scatter(psd_amp,psd_cf, alpha=.5)
plt.xlabel('AMP')
plt.ylabel('CF')
plt.show()

plt.scatter(psd_amp,psd_bw, alpha=.5)
plt.xlabel('AMP')
plt.ylabel('BW')
plt.show()

#%% HYPOTHESIS 2: THE SPECTRAL PARAMETERS AFFECT THE PAC VALUE

reg1 = linear_model.LinearRegression()
reg1.fit(psd_cf, pac_rhos)
xs = np.arange(min(psd_cf), max(psd_cf))
ys = reg1.intercept_[0] + reg1.coef_[0][0] * xs
plt.plot(xs, ys, '--k', linewidth=4, label='Model', alpha=.5)

plt.scatter(psd_cf, pac_rhos, alpha=.5)
plt.xlabel('CF')
plt.ylabel('Rho value')
plt.title('Higher Rho Values are only present in lower CFs')
plt.show()
# Higher Rho Values are only present in lower CFs. 


#%% HYPOTHESIS 3: SPECTRAL PARAMTERS & SIGNAL SYMMETRY

reg1 = linear_model.LinearRegression()
reg1.fit(psd_cf, median_pt_sym)
xs = np.arange(min(psd_cf), max(psd_cf))
ys = reg1.intercept_[0] + reg1.coef_[0][0] * xs
plt.plot(xs, ys, '--k', linewidth=4, label='Model', alpha=.5)


reg2 = linear_model.LinearRegression()
reg2.fit(psd_cf, median_rd_sym)
xs = np.arange(min(psd_cf), max(psd_cf))
ys = reg2.intercept_[0] + reg2.coef_[0][0] * xs
plt.plot(xs, ys, '--k', linewidth=4, label='Model', alpha=.5)

plt.scatter(psd_cf, median_pt_sym , alpha=.5)
plt.scatter(psd_cf, median_rd_sym, alpha=.5)
plt.xlabel('CF')
plt.ylabel('PT/RD symmetry')
plt.title('signals with higher CF have nearly perfect symmetry')
plt.show()
# signals with higher central frequencies have nearly perfect symmetry


reg1 = linear_model.LinearRegression()
reg1.fit(psd_bw, median_pt_sym)
xs = np.arange(min(psd_bw), max(psd_bw))
ys = reg1.intercept_[0] + reg1.coef_[0][0] * xs
plt.plot(xs, ys, '--k', linewidth=4, label='Model', alpha=.5)


reg2 = linear_model.LinearRegression()
reg2.fit(psd_bw, median_rd_sym)
xs = np.arange(min(psd_bw), max(psd_bw))
ys = reg2.intercept_[0] + reg2.coef_[0][0] * xs
plt.plot(xs, ys, '--k', linewidth=4, label='Model', alpha=.5)

plt.scatter(psd_bw, median_pt_sym, alpha=.5)
plt.scatter(psd_bw, median_rd_sym, alpha=.5)

plt.xlabel('BW')
plt.ylabel('PT/RD symmetry')
plt.title('Same for BW, higher ones have more symmetry')
plt.show()
# Same for BW, higher ones have more symmetry

#%% Clustering Symmetry with Rho data

plt.figure(figsize=(15,10))

# Blue Color: low PT and low Rho
plt.scatter(features_df["pac_rhos"][~(features_df["pac_rhos"] >.10) |  ~(features_df["median_pt_sym"] >.55)],
            features_df["median_pt_sym"][~(features_df["pac_rhos"] >.10) | ~(features_df["median_pt_sym"] >.55)], 
            c='blue', label='low PT low Rho')

# Red Color: low PT and high Rho 
plt.scatter(features_df["pac_rhos"][(features_df["pac_rhos"] >.10)],
            features_df["median_pt_sym"][(features_df["pac_rhos"] >.10)],
            c='red', label='low PT high Rho')

# Yellow Color: High PT low Rho
plt.scatter(features_df["pac_rhos"][(features_df["median_pt_sym"] >.55)],
            features_df["median_pt_sym"][(features_df["median_pt_sym"] >.55)], 
            c='yellow', label='High PT low Rho')

# Green Color: High PT high Rho
plt.scatter(features_df["pac_rhos"][(features_df["pac_rhos"] >.10) & (features_df["median_pt_sym"] >.55)],
            features_df["median_pt_sym"][(features_df["pac_rhos"] >.10) & (features_df["median_pt_sym"] >.55)], 
            c='green', label='High PT high Rho')


plt.xlabel('Rho Value')
plt.ylabel('PT sym')
plt.title('Rho values have a high PT symmetry. If they have a low PT they have a low Rho')
plt.legend(scatterpoints=1, loc='upper left');
plt.show()



#See where those datapoints are if we use them on the RD scatter

plt.figure(figsize=(15,10))

# Blue Color: low PT and low Rho
plt.scatter(features_df["pac_rhos"][~(features_df["pac_rhos"] >.10) |  ~(features_df["median_pt_sym"] >.55)],
            features_df["median_rd_sym"][~(features_df["pac_rhos"] >.10) | ~(features_df["median_pt_sym"] >.55)], 
            c='blue', label='low PT low Rho')

# Red Color: low PT and high Rho 
plt.scatter(features_df["pac_rhos"][(features_df["pac_rhos"] >.10)],
            features_df["median_rd_sym"][(features_df["pac_rhos"] >.10)],
            c='red', label='low PT high Rho')

# Yellow Color: High PT low Rho
plt.scatter(features_df["pac_rhos"][(features_df["median_pt_sym"] >.55)],
            features_df["median_rd_sym"][(features_df["median_pt_sym"] >.55)], 
            c='yellow', label='High PT low Rho')

# Green Color: High PT high Rho
plt.scatter(features_df["pac_rhos"][(features_df["pac_rhos"] >.10) & (features_df["median_pt_sym"] >.55)],
            features_df["median_rd_sym"][(features_df["pac_rhos"] >.10) & (features_df["median_pt_sym"] >.55)], 
            c='green', label='High PT high Rho')


plt.xlabel('Rho Value')
plt.ylabel('RD sym')
plt.title('Using same datapoint colors: Some of this structure also found in RD Sym, but less consistent')
plt.legend(scatterpoints=1, loc='upper left');
plt.show()


#See where those datapoints are when we use CF scatter

plt.figure(figsize=(15,10))

# Blue Color: low PT and low Rho
plt.scatter(features_df["pac_rhos"][~(features_df["pac_rhos"] >.10) |  ~(features_df["median_pt_sym"] >.55)],
            features_df["psd_cf"][~(features_df["pac_rhos"] >.10) | ~(features_df["median_pt_sym"] >.55)], 
            c='blue', label='low PT low Rho')

# Red Color: low PT and high Rho 
plt.scatter(features_df["pac_rhos"][(features_df["pac_rhos"] >.10)],
            features_df["psd_cf"][(features_df["pac_rhos"] >.10)],
            c='red', label='low PT high Rho')

# Yellow Color: High PT low Rho
plt.scatter(features_df["pac_rhos"][(features_df["median_pt_sym"] >.55)],
            features_df["psd_cf"][(features_df["median_pt_sym"] >.55)], 
            c='yellow', label='High PT low Rho')

# Green Color: High PT high Rho
plt.scatter(features_df["pac_rhos"][(features_df["pac_rhos"] >.10) & (features_df["median_pt_sym"] >.55)],
            features_df["psd_cf"][(features_df["pac_rhos"] >.10) & (features_df["median_pt_sym"] >.55)], 
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
plt.scatter(features_df["pac_rhos"][~(features_df["pac_rhos"] > rho_low) |  ~(features_df["resamp_zvals"] > zval_low)],
            features_df["resamp_zvals"][~(features_df["pac_rhos"] > rho_low) | ~(features_df["resamp_zvals"] > zval_low)], 
            c='blue', label='low Zval low Rho')

# Red Color: low PT and high Rho 
plt.scatter(features_df["pac_rhos"][(features_df["pac_rhos"] > rho_low)],
            features_df["resamp_zvals"][(features_df["pac_rhos"] > rho_low)],
            c='red', label='low Zval high Rho')

# Yellow Color: High PT low Rho
plt.scatter(features_df["pac_rhos"][(features_df["resamp_zvals"] > zval_low)],
            features_df["resamp_zvals"][(features_df["resamp_zvals"] > zval_low)], 
            c='yellow', label='High Zval low Rho')

# Green Color: High PT high Rho
plt.scatter(features_df["pac_rhos"][(features_df["pac_rhos"] > rho_hi) & (features_df["resamp_zvals"] > zval_hi)],
            features_df["resamp_zvals"][(features_df["pac_rhos"] > rho_hi) & (features_df["resamp_zvals"] > zval_hi)], 
            c='green', label='High Zval high Rho')

plt.xlabel('Rho Value')
plt.ylabel('Resamp Z-value')
plt.title('Categorizing the outcome measures, although they are strongly related')
plt.legend(scatterpoints=1, loc='upper left');
plt.show()

#%% The outcome measures/statistics are somewhat related to the Symmetry measures

plt.figure(figsize=(15,10))

# Blue Color: low Zval and low Rho
plt.scatter(features_df["median_pt_sym"][~(features_df["pac_rhos"] > rho_low) |  ~(features_df["resamp_zvals"] > zval_low)],
            features_df["median_rd_sym"][~(features_df["pac_rhos"] > rho_low) | ~(features_df["resamp_zvals"] > zval_low)], 
            c='blue', label='low Zval low Rho')

# Red Color: low PT and high Rho 
plt.scatter(features_df["median_pt_sym"][(features_df["pac_rhos"] > rho_low)],
            features_df["median_rd_sym"][(features_df["pac_rhos"] > rho_low)],
            c='red', label='low Zval high Rho')

# Yellow Color: High PT low Rho
plt.scatter(features_df["median_pt_sym"][(features_df["resamp_zvals"] > zval_low)],
            features_df["median_rd_sym"][(features_df["resamp_zvals"] > zval_low)], 
            c='yellow', label='High Zval low Rho')

# Green Color: High PT high Rho
plt.scatter(features_df["median_pt_sym"][(features_df["pac_rhos"] > rho_hi) & (features_df["resamp_zvals"] > zval_hi)],
            features_df["median_rd_sym"][(features_df["pac_rhos"] > rho_hi) & (features_df["resamp_zvals"] > zval_hi)], 
            c='green', label='High Zval high Rho')

plt.xlabel('PT Symmetry')
plt.ylabel('RD Symmetry')
plt.title('The outcome measures/statistics are somewhat related to the Symmetry measures')
plt.legend(scatterpoints=1, loc='upper left');
plt.show()

#%% High outcome measures have a high PT sym and a low CF

plt.figure(figsize=(15,10))

# Blue Color: low Zval and low Rho
plt.scatter(features_df["median_pt_sym"][~(features_df["pac_rhos"] > rho_low) |  ~(features_df["resamp_zvals"] > zval_low)],
            features_df["psd_cf"][~(features_df["pac_rhos"] > rho_low) | ~(features_df["resamp_zvals"] > zval_low)], 
            c='blue', label='low Zval low Rho')

# Red Color: low PT and high Rho 
plt.scatter(features_df["median_pt_sym"][(features_df["pac_rhos"] > rho_low)],
            features_df["psd_cf"][(features_df["pac_rhos"] > rho_low)],
            c='red', label='low Zval high Rho')

# Yellow Color: High PT low Rho
plt.scatter(features_df["median_pt_sym"][(features_df["resamp_zvals"] > zval_low)],
            features_df["psd_cf"][(features_df["resamp_zvals"] > zval_low)], 
            c='yellow', label='High Zval low Rho')

# Green Color: High PT high Rho
plt.scatter(features_df["median_pt_sym"][(features_df["pac_rhos"] > rho_hi) & (features_df["resamp_zvals"] > zval_hi)],
            features_df["psd_cf"][(features_df["pac_rhos"] > rho_hi) & (features_df["resamp_zvals"] > zval_hi)], 
            c='green', label='High Zval high Rho')

plt.xlabel('PT Symmetry')
plt.ylabel('CF')
plt.title('High outcome measures have a high PT sym and a low CF')
plt.legend(scatterpoints=1, loc='upper left');
plt.show()

#%% Higher outcome measures have less variation in symmetry (especially in RD)

plt.figure(figsize=(15,10))

# Blue Color: low Zval and low Rho
plt.scatter(features_df["std_rd_sym"][~(features_df["pac_rhos"] > rho_low) |  ~(features_df["resamp_zvals"] > zval_low)],
            features_df["pac_rhos"][~(features_df["pac_rhos"] > rho_low) | ~(features_df["resamp_zvals"] > zval_low)], 
            c='blue', label='low Zval low Rho')

# Red Color: low PT and high Rho 
plt.scatter(features_df["std_rd_sym"][(features_df["pac_rhos"] > rho_low)],
            features_df["pac_rhos"][(features_df["pac_rhos"] > rho_low)],
            c='red', label='low Zval high Rho')

# Yellow Color: High PT low Rho
plt.scatter(features_df["std_rd_sym"][(features_df["resamp_zvals"] > zval_low)],
            features_df["pac_rhos"][(features_df["resamp_zvals"] > zval_low)], 
            c='yellow', label='High Zval low Rho')

# Green Color: High PT high Rho
plt.scatter(features_df["std_rd_sym"][(features_df["pac_rhos"] > rho_hi) & (features_df["resamp_zvals"] > zval_hi)],
            features_df["pac_rhos"][(features_df["pac_rhos"] > rho_hi) & (features_df["resamp_zvals"] > zval_hi)], 
            c='green', label='High Zval high Rho')

plt.xlabel('RD STD')
plt.ylabel('Rho Value')
plt.title('High outcome measures have less variation in the RD symmetry')
plt.legend(scatterpoints=1, loc='upper left');
plt.show()

#%% Higher outcome measures have less variation in symmetry (especially in RD)

plt.figure(figsize=(15,10))

# Blue Color: low Zval and low Rho
plt.scatter(features_df["std_rd_sym"][~(features_df["pac_rhos"] > rho_low) |  ~(features_df["resamp_zvals"] > zval_low)],
            features_df["median_pt_sym"][~(features_df["pac_rhos"] > rho_low) | ~(features_df["resamp_zvals"] > zval_low)], 
            c='blue', label='low Zval low Rho')

# Red Color: low PT and high Rho 
plt.scatter(features_df["std_rd_sym"][(features_df["pac_rhos"] > rho_low)],
            features_df["median_pt_sym"][(features_df["pac_rhos"] > rho_low)],
            c='red', label='low Zval high Rho')

# Yellow Color: High PT low Rho
plt.scatter(features_df["std_rd_sym"][(features_df["resamp_zvals"] > zval_low)],
            features_df["median_pt_sym"][(features_df["resamp_zvals"] > zval_low)], 
            c='yellow', label='High Zval low Rho')

# Green Color: High PT high Rho
plt.scatter(features_df["std_rd_sym"][(features_df["pac_rhos"] > rho_hi) & (features_df["resamp_zvals"] > zval_hi)],
            features_df["median_pt_sym"][(features_df["pac_rhos"] > rho_hi) & (features_df["resamp_zvals"] > zval_hi)], 
            c='green', label='High Zval high Rho')

plt.xlabel('RD STD')
plt.ylabel('median_pt_sym')
plt.title('High outcome measures have less variation in the RD symmetry')
plt.legend(scatterpoints=1, loc='upper left');
plt.show()

#%% NEXT STEPS: 

# 1) get all features
# 2) look at colloniarity (correlations) and select features
# 3) do PCA
# 4) unsupervised clustering



#%% Inspect features

pd.plotting.scatter_matrix(features_df, figsize=(40,40))


#%%


f = plt.figure(figsize=(19, 15))
plt.matshow(features_df.corr(), fignum=f.number)
plt.xticks(range(features_df.shape[1]), features_df.columns, fontsize=14, rotation=45)
plt.yticks(range(features_df.shape[1]), features_df.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16);

#%% adjust features in features_df

# drop backgr_knee, because no correlations with other data, 
# and has homogeneous data

features_df = features_df.drop(columns='backgr_knee')

# Log transfer psd_bw
features_df['psd_bw_log'] = np.log10(features_df['psd_bw'])
# drop old psd_bw for clean df
features_df = features_df.drop(columns='psd_bw')

# and create 1 feature out of pac_rhos + resamp_zvals
# and create 1 feature out of backgr_exp and offset


#%% Scale data & PCA example

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import mglearn


#%% Set data to right structure for ML with sklearn

# first, logtransform bandwidth
psd_bw_log10 = np.log10(psd_bw)

# change to right shape for sklearn
pac_rhos = np.reshape(pac_rhos, [len(pac_rhos), 1])
resamp_zvals = np.reshape(resamp_zvals, [len(resamp_zvals), 1])
median_rd_sym  = np.reshape(median_rd_sym, [len(median_rd_sym), 1])
median_pt_sym = np.reshape(median_pt_sym, [len(median_pt_sym), 1])
psd_cf = np.reshape(psd_cf, [len(psd_cf), 1])
psd_amp = np.reshape(psd_amp, [len(psd_amp), 1])
backgr_exp = np.reshape(backgr_exp, [len(backgr_exp), 1])
backgr_offset = np.reshape(backgr_offset, [len(backgr_offset), 1])
median_volt_amp = np.reshape(median_volt_amp, [len(median_volt_amp), 1])
psd_bw_log10 = np.reshape(psd_bw_log10, [len(psd_bw_log10), 1])

# and create 1 feature out of pac_rhos + resamp_zvals
scaler = StandardScaler()
pac_values = scaler.fit_transform(pac_rhos) + scaler.fit_transform(resamp_zvals)

# and create 1 feature out of backgr_exp and offset
scaler = StandardScaler()
aperiodic_param = scaler.fit_transform(backgr_exp) + scaler.fit_transform(backgr_offset)

# change to right shape for sklearn 
aperiodic_param = np.reshape(aperiodic_param, [len(aperiodic_param), 1])
pac_values = np.reshape(pac_values, [len(pac_values), 1])

pac_features = np.hstack((pac_values, median_rd_sym, median_pt_sym, median_volt_amp, \
                           aperiodic_param, psd_cf, psd_amp, psd_bw_log10))

# scale data
scaler = StandardScaler()
scaler.fit(pac_features)
X_scaled = scaler.transform(pac_features)

#%% PAC Scale data & PCA 


feature_list = ['pac_values', 'median_rd_sym', 'median_pt_sym', 'median_volt_amp', \
                'aperiodic_param', 'psd_cf', 'psd_amp', 'psd_bw_log10']

# PCA
pca = PCA(n_components=2)
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)

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
plt.xticks(range(len(feature_list)),feature_list, rotation=60, ha='left')
plt.xlabel("Feature")
plt.ylabel("Principal components")


#%% K-means

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
kmeans.fit(X_scaled)

# visualize on PCA  
mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], kmeans.labels_, markers='o')
mglearn.discrete_scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], [0, 1], markers='^', markeredgewidth=3)
plt.title('K-Means: Plot clusters visualized on PCA\'s including center')
plt.xlabel("First principal component")
plt.ylabel("Second principal component")

features_df['Clusters'] = kmeans.labels_
cluster_label = kmeans.labels_

#%% Agglomerative clustering

from sklearn.cluster import AgglomerativeClustering
agg = AgglomerativeClustering(n_clusters=2)
assignment = agg.fit_predict(X_scaled)
mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], assignment)
plt.title('AggloClustering: Plot clusters visualized on PCA\'s') 
plt.xlabel("Component 1")
plt.ylabel("Component 2")


#%% Hierarchical cluster / Dendogram

from scipy.cluster.hierarchy import dendrogram, ward

linkage_array = ward(X_scaled)

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
clusters = dbscan.fit_predict(X_scaled)
# plot the cluster assignments
plt.scatter(X_scaled[:, 0], X_pca[:, 1])#, c=clusters, cmap=mglearn.cm2, s=60)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")


#%% Linear regression predicting pac_values in each cluster

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
kmeans.fit(X_scaled)

# or without PAC_values included
# kmeans.fit(X_scaled[:,1:])

# visualize on PCA  
mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], kmeans.labels_, markers='o')
mglearn.discrete_scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], [0, 1], markers='^', markeredgewidth=3)
plt.title('K-Means: Plot clusters visualized on PCA\'s including center')
plt.xlabel("First principal component")
plt.ylabel("Second principal component")

features_df['Clusters'] = kmeans.labels_
cluster_label = kmeans.labels_


# which channels/data are in which cluster
features_cluster0 = np.zeros([(cluster_label==0).sum(),len(X_scaled[0])])
features_cluster1 = np.zeros([(cluster_label==1).sum(),len(X_scaled[0])])

for ii in range(len(X_scaled[0])):
    features_cluster0[:,ii] = [X_scaled[jj,ii] for jj in range(len(X_scaled)) if cluster_label[jj] == 0]
    features_cluster1[:,ii] = [X_scaled[jj,ii] for jj in range(len(X_scaled)) if cluster_label[jj] == 1]
    
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
plt.xticks(index + bar_width, feature_list[1:8],fontsize=16)
plt.legend(fontsize=16)

#plt.tight_layout()
plt.show()



# get differences in beta values



#%%% Plot difference in features between NoPacs and PACs in histograms

# manually set which columns you want to plot
plot_list = [0,1,2,3,4,5,6,8,9]



plt.figure(figsize=(20,20))
for ii in range(len(plot_list)): 
        
    jj = plot_list[ii]
    # subplots 2x5
    plt.subplot(3,3,ii+1)
    #xticks([]), yticks([])
    plt.title(features_df.columns[jj])
    plt.hist(features_df_nopac.iloc[:,jj], bins=20, alpha=.4)
    plt.axvline(np.median(features_df_nopac.iloc[:,jj]), color='b', linestyle='dashed', linewidth=1)
    
    plt.hist(features_df.iloc[:,jj], bins=20, alpha=.4)
    plt.axvline(np.median(features_df.iloc[:,jj]), color='r', linestyle='dashed', linewidth=1)
    

#%% Linear regression beta values plotted that predict the Rho/Z-value
### Including: cluster 0, cluster 1, and the NoPAC group
    

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
kmeans.fit(X_scaled)

# or without PAC_values included
# kmeans.fit(X_scaled[:,1:])

# visualize on PCA  
mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], kmeans.labels_, markers='o')
mglearn.discrete_scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], [0, 1], markers='^', markeredgewidth=3)
plt.title('K-Means: Plot clusters visualized on PCA\'s including center')
plt.xlabel("First principal component")
plt.ylabel("Second principal component")

features_df['Clusters'] = kmeans.labels_
cluster_label = kmeans.labels_


# which channels/data are in which cluster
features_cluster0 = np.zeros([(cluster_label==0).sum(),len(X_scaled[0])])
features_cluster1 = np.zeros([(cluster_label==1).sum(),len(X_scaled[0])])

for ii in range(len(X_scaled[0])):
    features_cluster0[:,ii] = [X_scaled[jj,ii] for jj in range(len(X_scaled)) if cluster_label[jj] == 0]
    features_cluster1[:,ii] = [X_scaled[jj,ii] for jj in range(len(X_scaled)) if cluster_label[jj] == 1]
    
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


# get X_scaled_nopac
X_nopac = X_scaled_nopac[:,1:]
Y_nopac = X_scaled_nopac[:,0]

# linear regression
reg_nopac = linear_model.LinearRegression().fit(X_nopac, Y_nopac)


# create plot
#fig, ax = plt.subplots()
plt.figure(figsize=(20,8))
index = np.arange(7)
bar_width = 0.25

rects1 = plt.bar(index, reg_cluster0.coef_, bar_width,
color='b',
label='Cluster 0')

rects2 = plt.bar(index + bar_width, reg_cluster1.coef_, bar_width,
color='r',
label='Cluster 1')


rects3 = plt.bar(index + (bar_width * 2), reg_nopac.coef_, bar_width,
color='g',
label='No PAC')

plt.xlabel('Feature')
plt.ylabel('Beta Value',fontsize=16)
plt.title('Linear regression: features on PAC-value',fontsize=16)
plt.xticks(index + bar_width, feature_list[1:8],fontsize=16)
plt.legend(fontsize=16)

#plt.tight_layout()
plt.show()


