import os
import scipy.io as sio
import glob
import numpy as np


#%% 
    
def load_data_human(dat_name, subjects, fs, timewindow):
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
       

    return datastruct, elec_locs

#%% 
  
def load_data_monkey(eyes_closed, subjects, fs, epoch_len, num_epoch, channels):
    
    """ This function load the ECoG data from monkeys from neurotycho. Set the link 
    to the right data map manually in the function because it has to return to this 
    map every new subject. 
    
    Give following inputs:
    -   time when eyes_closed paradigm starts
    -   list of subjects 
    -   sampling frequency
    -   length of epochs and number of epochs you want to derive
    -   number of channels
    """ 
    
    datastruct = [None] * len(subjects)
    
    for subj in range(len(subjects)): 
           
        # go to specific map
        os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab\neurotycho\anesthesia_sleep_task')    
        filename = os.path.join(os.getcwd(), subjects[subj])   
        os.chdir(filename)
    
        ch_counter = 0 
        datastruct_ch = [None] * channels
    
        for file_idx in glob.glob("ECoG_ch*.mat"):
            
            # get data
            data = sio.loadmat(file_idx)
            data = next(v for k,v in data.items() if 'ECoGData' in k)
            
            # get only eyes_closed part of data
            data = np.squeeze(data)[
                    int(eyes_closed[subj][0] * fs):int(eyes_closed[subj][1] * fs)
                    ]
            
            # for every epoch of 20 seconds, write data to channel specific structure
            datastruct_ch[ch_counter] = [data[ep * epoch_len:ep * epoch_len + epoch_len] for ep in range(num_epoch)]
            
            # counter
            ch_counter = ch_counter + 1 
        
        # write data to full structure
        datastruct[subj] = datastruct_ch
        
    return datastruct

#%%
 
def load_data_rat(subjects, fs, epoch_len_seconds, num_epoch, num_tetrodes):   

    """ This function load the hippocampal recordings of rats from CRCNS. 
    Set the link to the right data map manually in the function because it 
    has to return to this map every new subject. Function searches in task map 
    for the first resting state task and goes to the according map.
    
    Give following inputs:
    -   list of subjects 
    -   sampling frequency
    -   length of epochs and number of epochs you want to derive
    -   number of tetrodes
    """ 
    
    # length epoch
    epoch_len = int(epoch_len_seconds * fs)
    
    # create datastructure
    datastruct = [None] * len(subjects)
    
    # for every subject
    for subj in range(len(subjects)): 
    
        # go to task map
        os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab\hippocampus_rats')    
        task_dir = os.path.join(os.getcwd(), subjects[subj])
        os.chdir(task_dir)
        
        # day specific datastructure
        datastruct_day = [None] * len(glob.glob("*task*.mat"))
        
        for day_counter in range(len(glob.glob("*task*.mat"))):
            
            # go to task map again
            os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab\hippocampus_rats')    
            task_dir = os.path.join(os.getcwd(), subjects[subj])
            os.chdir(task_dir)
        
            # get task file of the first recording day
            task_idx = glob.glob("*task*.mat")[day_counter]
            task = sio.loadmat(task_idx)
            
            # get day of the task
            day = task_idx.split('task')[1].split('.mat')[0]
            
            # if first task is sleep task
            if task['task'][0,int(day)-1][0,0][0,0][0][0] == 'sleep': 
                
                # go to the subject folder with the data
                os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab\hippocampus_rats')  
                filename = os.path.join(os.getcwd(), subjects[subj], 'EEG')   
                os.chdir(filename)
        
                
                # channel datastruct with length of number of channels    
                datastruct_ch = [None] * num_tetrodes
            
                # for each recording for that day
                for file_idx in glob.glob(('*eeg' + day)+'-1-*.mat'):
                    
                    # get data
                    data = sio.loadmat(file_idx)
                    
                    # find channel by splitting file name -1 for index
                    ch = int(file_idx.split('eeg' + day + '-1-')[1].split('.mat')[0]) - 1
                    
                    # get fs and start_time
                    # fs = data['eeg'][0,int(day)-1][0,0][0,ch][0,0]['samprate'][0][0]
                    # start_time = data['eeg'][0,int(day)-1][0,ep][0,ch_counter][0,0]['starttime'][0][0]     
                    
                    # squeeze data
                    data = np.squeeze(data['eeg'][0,int(day)-1][0,0][0,ch][0,0]['data'])
        
                    
                    # for every epoch of 20 seconds, write data to channel specific structure
                    datastruct_ch[ch] = [data[ep * epoch_len:ep * epoch_len + epoch_len] for ep in range(num_epoch)]
    
                
            
            # write channel data to day structure
            datastruct_day[day_counter] = datastruct_ch
    
                
        # write data to full structure
        datastruct[subj] = datastruct_day
        
    return datastruct
