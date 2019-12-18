# find best parameters for FOOOF

freq_range = [4, 58]
bw_lims = [2, 8]
max_n_peaks = 4  

for ii in range(0,10):

    ch = ii 
    
    # get signal
    sig = datastruct[subj][ch]
    
    # compute frequency spectrum
    freq_mean, psd_mean = spectral.compute_spectrum(sig, fs, method='welch', avg_type='mean', nperseg=fs*2)
    
    
    if sum(psd_mean) == 0: 
        
        peak_params = np.empty([0, 3])
        
        psd_peak_chs.append(peak_params)
        
    else:
        
        # Initialize FOOOF model
        fm = FOOOF(peak_width_limits=bw_lims, background_mode='knee', max_n_peaks=max_n_peaks)
        
        # fit model
        fm.fit(freq_mean, psd_mean, freq_range)
        fm.report()



#%% 

import matplotlib.pyplot as plt
from neurodsp import spectral
from neurodsp.plts.spectral import plot_power_spectra


# get signal
sig = datastruct[0][0]
fs = 1000
plt_time = [60000, 62000]



#%%

def add_titlebox(ax, text, qq):
    ax.text(.55, .8 + (qq/10), text,
            horizontalalignment='center',
            verticalalignment='bottom',
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.3),
            fontsize=16.0)
    return ax

gridsize = (3, 2)
fig = plt.figure(figsize=(20, 20))
ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=2, rowspan=1)
ax2 = plt.subplot2grid(gridsize, (1, 0))
ax3 = plt.subplot2grid(gridsize, (1, 1))
ax4 = plt.subplot2grid(gridsize, (2, 0))
ax5 = plt.subplot2grid(gridsize, (2, 1))


qq = 0

######## ax2: PSD Welch
freq_mean, psd_mean = spectral.compute_spectrum(sig, fs, method='welch', avg_type='mean', nperseg=fs*2)
ax2.loglog(freq_mean[:118],psd_mean[:118])
add_titlebox(ax2, 'PSD Welch up to 118 Hz', 1)

######## ax5: stats 

# Initialize FOOOF model
fm = FOOOF(peak_width_limits=bw_lims, background_mode='knee', max_n_peaks=max_n_peaks)

# fit model
fm.fit(freq_mean, psd_mean, freq_range) 

# Central frequency, Amplitude, Bandwidth
peak_params = fm.peak_params_

#offset, knee, slope
background_params = fm.background_params_

# find which peak has the biggest amplitude
max_ampl_idx = np.argmax(peak_params[:,1])
    
# define biggest peak in power spectrum and add to channel array
peak_params = peak_params[max_ampl_idx]

# periodics
if len(peak_params) > 0:
    if (peak_params[0] < 15) & (peak_params[1] >.2) & (peak_params[1] < 1.5): 
        
        add_titlebox(ax5, 'Central Frequency' + '    ' + str(peak_params[0]), -2)
        add_titlebox(ax5, 'Amplitude' + '    ' + str(peak_params[1]), -3)
        add_titlebox(ax5, 'Bandwidth' + '    ' + str(peak_params[2]), -4)
    
else: 
    add_titlebox(ax5, 'No true oscillation < 15 Hz present', -2)

# aperiodics
add_titlebox(ax5, 'Offset:' + '    ' +  str(background_params[0]), 0)
add_titlebox(ax5, 'Slope' + '    ' + str(background_params[2]), 1)

# PAC values
#add_titlebox(ax5, 'Offset:' + '    ' +  str(background_params[0]), 0)
#add_titlebox(ax5, 'Slope' + '    ' + str(background_params[2]), 1)


####### ax1: Signal
if 
ax1.plot((sig[plt_time[0]:plt_time[1]]),label= 'Raw Signal',color='black')
add_titlebox(ax1, '2 seconds of signal')
