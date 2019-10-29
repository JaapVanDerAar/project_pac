


#%% Welch frequency distribution
from neurodsp.plts.spectral import plot_power_spectra
plot_power_spectra(freq_mean[:200], psd_mean[:200], 'Welch')


#%% Simple plot 

from neurodsp.plts.time_series import plot_time_series

times = create_times(len(datastruct[subj][ch])/fs, fs)
plot_time_series(times, datastruct[subj][ch], xlim=[0, .8])

#%%
with pd.option_context('display.max_rows', 20
                       , 'display.max_columns', None):    
    print(df)

#%%