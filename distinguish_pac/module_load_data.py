import os
import scipy.io as sio


import time

        


#%% 
def load_data_and_locs(dat_name, subjects, fs, timewindow, ss):
    """ This function load the ECoG data from Kai Miller's database 
    Give following inputs:
    -   name of data
    -   list of subjects 
    -   sampling frequency
    -   timewindow of which data to include
    -   starting point from where to extract data (in seconds) """ 
    
    # calculate starting sample and total sample time
    ss = ss * fs
    tw = timewindow * fs
       
    # create empty output structures
    datastruct = [None] * len(subjects) 
    elec_locs = [None] * len(subjects)
        
    for subj in range(len(subjects)):
        
        # get the filename
        sub_label = subjects[subj] + '_base'
        filename = os.path.join(os.getcwd(), dat_name, 'data', sub_label)
    
        # load data
        dataStruct = sio.loadmat(filename)
        
        # get electrodes
        elec_locs[subj] = dataStruct['locs']
        
              
        # get specific part of data of each channel            
        data = dataStruct['data']
        datastruct[subj] = [data[ss:ss+tw,ch] for ch in range(len(data[0]))]
        
    return datastruct, elec_locs

#%% 
    
def load_data_timewindow(dat_name, subjects, fs, timewindow):
    """ This function load the ECoG data from Kai Miller's database 
    Difference with other load function is that this function takes the 
    timewindow of the subject with smallest data and takes middle part of data in 
    other subjects 
    
    Give following inputs:
    -   name of data
    -   list of subjects 
    -   sampling frequency
    -   timewindow of which data to include
    """ 
  
    start = time.time()
    print("hello")
    
    datastruct = [None] * len(subjects) 
    elec_locs = [None] * len(subjects)
        
    for subj in range(len(subjects)):
        
        # get the filename
        sub_label = subjects[subj] + '_base'
        filename = os.path.join(os.getcwd(), dat_name, 'data', sub_label)
    
        # load data
        dataStruct = sio.loadmat(filename)
        data = dataStruct['data']
        locs = dataStruct['locs']
            
        # set time parameters
        ss = round((len(data)/2)/fs) - (timewindow/2) 
    
        ss = int(ss * fs)
        
        tw = int(timewindow * fs)
        
    
        # get specific part of data of each channel  
        datastruct[subj] = [data[ss:ss+tw,ch] for ch in range(len(data[0]))]
      
        # save electrode locations        
        elec_locs[subj] = locs
       
    
    end = time.time()
    print(end - start)    

    return datastruct, elec_locs
