import os
import numpy as np
import scipy.io as sio


import glob
import pandas as pd
import matplotlib.pyplot as plt

#%% Load variables to put in spatial map
        
# Set directory in which the data structure can be found
os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab')

# dataframe
features_df = pd.read_csv('features_df.csv', sep=',')  

subjects = ['Chibi', 'George', 'Kin2', 'Su']


#%% Function
# STILL NEED THE VARIABLES TO INCLUDE 

def spatial_map_neurotycho(subjects, variable):
    
    # set vmin and vmax as quantiles to deal with outliers and make consistent labeling
    vmin = features_df[var].quantile(0.05)
    vmax = features_df[var].quantile(0.95)
       
    for subj in range(len(subjects)):  
        # go to specific map
        os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab\neurotycho\anesthesia_sleep_task')    
        filemap = os.path.join(os.getcwd(), subjects[subj])   
        os.chdir(filemap)
        filename = glob.glob("*Map.mat")
        spatial_map = sio.loadmat(filename[0])
        
        var_values = [
                features_df[(features_df['subj'] == subj) & (features_df['ch'] == ch)][var].median() 
                for ch in range(len(spatial_map['X']))
                    ] 
        
        var_values = np.reshape(var_values, [len(var_values), 1]) 
        
        plt.figure(figsize=(10,10))
        plt.imshow(spatial_map['I'])
        plt.scatter(spatial_map['X'], spatial_map['Y'], s=100, c=var_values, cmap='coolwarm',
                    vmin=vmin, vmax=vmax)
        plt.title(subjects[subj], size=20)
        plt.axis('off')
        cbar= plt.colorbar()
        cbar.set_label(var, size=20)
        plt.show()


#%% Which variable to use in function and run it
        
var = 'resamp_pac_zvals'

spatial_map_neurotycho(subjects, var)
