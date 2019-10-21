# import stuff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import patsy
import statsmodels.api as sm

#%% correlate some data

corr = 0.75
covs = [[1, corr], [corr, 1]] 
mean = [0, 0]

dat = np.random.multivariate_normal(mean, covs, 1000)


#%% Scatter show data

plt.scatter(dat[:,0], dat[:,1], alpha = 0.3)


#%% Put into DataFrame

df = pd.DataFrame(dat, columns=['D1', 'D2'])


#%% Patsy - for easy way to onstruct design matrices
### just a way to organize the matrices for predictor and output variables

outcome, predictors = patsy.dmatrices('D2 ~ D1', df)

#%% Statsmodels

# initialize OLS linear model and provides the data, but not actually computes it.
# uses a OOP approach, whereby complex objects are initialized that stor
# the data and methods together
mod = sm.OLS(outcome, predictors) 

# fit the model
res = mod.fit()

# check summary
print(res.summary())

#%% Plot the data and the model 

plt.scatter(df['D1'], df['D2'], alpha=0.3, label='Data')

# give x coordinates
xs = np.arange(df['D1'].min(),df['D1'].max())

# get y coordinates

# first, extract those from the statsmodels summary
res_summary = res.summary()
results_as_html = res_summary.tables[1].as_html()
results = pd.read_html(results_as_html, header =0, index_col=0)[0]

# put coefficients in for y data
ys = results['coef'][0] + results['coef'][1] * xs


plt.plot(xs, ys, '--k', linewidth=3, label='Model')



#%% Add another predictor and predict

df['D3'] = pd.Series(np.random.randn(1000), index=df.index)

outcome, predictors = patsy.dmatrices('D1 ~ D2 + D3', df)
mod = sm.OLS(outcome, predictors)
res = mod.fit()

#%% Using SKlearn

from sklearn import linear_model

#%% convert data to right shape

d1 = np.reshape(df.D1.values, [len(df.D1), 1])
d2 = np.reshape(df.D2.values, [len(df.D2), 1])
d3 = np.reshape(df.D3.values, [len(df.D3), 1])

#%% initialize linear regression model

reg = linear_model.LinearRegression()

reg.fit(d2, d1)

#%% check results - same as statsmodels

print(reg.intercept_[0])
print(reg.coef_[0][0])

#%% Sklearn with multiple predictors

reg = linear_model.LinearRegression()
reg.fit(np.hstack([d2, d3]), d1) 

#%% check and compare results

print('intercept: \t', reg.intercept_[0])
print('Thata D2 :\t', reg.coef_[0][0])
print('Thata D3 :\t', reg.coef_[0][1])





