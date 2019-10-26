


#%% Welch frequency distribution
from neurodsp.plts.spectral import plot_power_spectra
plot_power_spectra(freq_mean[:200], psd_mean[:200], 'Welch')


#%% Simple plot 

from neurodsp.plts.time_series import plot_time_series

times = create_times(len(sig)/fs, fs)
plot_time_series(times, sig, xlim=[0, 3])