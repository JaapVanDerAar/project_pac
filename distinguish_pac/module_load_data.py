import os
import scipy.io as sio

#%% 
def load_data_and_locs(dat_name, subjects, fs, timewindow, ss):
    """ This function load the ECoG data from Kai Miller's database 
    Give following inputs:
    -   name of data
    -   list of subjects 
    -   sampling frequency
    -   timewindow of which data to include
    -   starting point from where to extract data (in seconds) """ 
    
    ss = ss * fs
    
    tw = timewindow * fs
    

    datastruct = [] 
    elec_locs = []
        
    for subj in range(len(subjects)):
        
        # get the filename
        sub_label = subjects[subj] + '_base'
        filename = os.path.join(os.getcwd(), dat_name, 'data', sub_label)
    
        # load data
        dataStruct = sio.loadmat(filename)
        data = dataStruct['data']
        locs = dataStruct['locs']
    
    
        # create empty list for subj data
        subj_data = []
        
        
        # for every channel in this subj
        for ch in range(len(data[0])):
            
            # extract data of specified timewindow
            ch_data = data[ss:ss+tw,ch]
            
            # store channel data in sub_data
            subj_data.append(ch_data)
            
        
        datastruct.append(subj_data)
        
        elec_locs.append(locs)
        

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
    
    datastruct = [] 
    elec_locs = []
        
    for subj in range(len(subjects)):
        
        # get the filename
        sub_label = subjects[subj] + '_base'
        filename = os.path.join(os.getcwd(), dat_name, 'data', sub_label)
    
        # load data
        dataStruct = sio.loadmat(filename)
        data = dataStruct['data']
        locs = dataStruct['locs']
    
    
        # create empty list for subj data
        subj_data = []
        
        # set time parameters
        ss = round((len(data)/2)/fs) - (timewindow/2) 
    
        ss = int(ss * fs)
        
        tw = int(timewindow * fs)
        
        
        # for every channel in this subj
        for ch in range(len(data[0])):
            
            # extract data of specified timewindow
            ch_data = data[ss:ss+tw,ch]
            
            # store channel data in sub_data
            subj_data.append(ch_data)
            
        
        datastruct.append(subj_data)
        
        elec_locs.append(locs)
        

    return datastruct, elec_locs
