# function to load data in structure

subjects=['al','ca','cc','de','fp','gc','gf','gw',
          'h0','hh','jc','jm','jp','mv','rh','rr',
          'ug','wc','wm','zt']

os.chdir(r'C:\Users\jaapv\Desktop\master\VoytekLab')

dataset = 'fixation_pwrlaw'


def load_database(subjects):
    
    for subj in range(len(subjects)):
        
        # get the filename
        sub_label = subjects[subj] + '_base'
        filename = os.path.join(os.getcwd(), dataset, 'data', sub_label)

        # load data
        dataStruct = sio.loadmat(filename)
        data = dataStruct['data']
        locs = dataStruct['locs']
        
        
#%% 
    
import os
import scipy as sp

    

for subj in range(len(subjects)):
    
    # get the filename
    sub_label = subjects[subj] + '_base'
    filename = os.path.join(os.getcwd(), dataset, 'data', sub_label)
    
    # load data
    dataStruct = sp.io.loadmat(filename)
    data = dataStruct['data']
    locs = dataStruct['locs']       
        
    data = np.ndarray.tolist(data[10000:100000])
    
    
    
    database('ez') = data
    
    



