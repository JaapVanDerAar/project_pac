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

#%%

pd.plotting.scatter_matrix(features_df, figsize=(40,40))

#%%

plt.matshow(features_df.corr())

#%% NEXT STEPS: 

# 1) get all features
# 2) look at colloniarity (correlations) and select features
# 3) do PCA
# 4) unsupervised clustering










#%% ML - Supervised


# Prediction: Rho value
# Features: PT, RD, STD's, CF, BW

# pac_rhos = np.reshape(pac_rhos, [len(pac_rhos), 1])
psd_cf = np.reshape(psd_cf, [len(psd_cf), 1])
psd_bw = np.reshape(psd_bw, [len(psd_bw), 1])
median_pt_sym = np.reshape(median_pt_sym, [len(median_pt_sym), 1])
median_rd_sym  = np.reshape(median_rd_sym, [len(median_rd_sym), 1])
std_rd_sym  = np.reshape(std_rd_sym, [len(std_rd_sym), 1])
std_pt_sym  = np.reshape(std_pt_sym, [len(std_rd_sym), 1])

pac_rhos_binary = []
for ii in range(len(pac_rhos)):
    if pac_rhos[ii] > 0.1:
        pac_rhos_binary_0 = 1
    else: 
        pac_rhos_binary_0 = 0
        
    pac_rhos_binary.append(pac_rhos_binary_0)
        
pac_rhos_binary = np.reshape(pac_rhos_binary, [len(pac_rhos_binary)])

x_hat = np.hstack((median_pt_sym, median_rd_sym, std_pt_sym, std_rd_sym, psd_cf, psd_bw))
y_hat = pac_rhos_binary

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_hat,y_hat)

#%% ML - Unsupervised
# first scale features
