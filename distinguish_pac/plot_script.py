


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

import matplotlib.pyplot as plt




plt_time = [0, 5000]

plt.figure(figsize = (20,8));
plt.plot((sig[plt_time[0]:plt_time[1]]),label= 'Raw Signal')
plt.plot((amp_data_hilbert[plt_time[0]:plt_time[1]]),label= 'High Gamma [80-125 Hz]')
plt.plot((phase_data_hilbert[plt_time[0]:plt_time[1]]),label= 'Phase [{0:.2f} - {1:.2f} Hz]'.format(phase_providing_band[0], phase_providing_band[1]))
plt.plot((amp_data_abs[plt_time[0]:plt_time[1]]),label= 'Amplitude Envelope')
plt.plot((phase_data_angle[plt_time[0]:plt_time[1]]*0.1),label= 'Phase Angle')
plt.xlabel('subj: {0:.0f}, ch {1:.0f},  Two Seconds of Theta Phase, High Gamma Amplitude, and Raw Signal '.format(subj,ch))
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))