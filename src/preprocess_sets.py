import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import mne
import os
import time
from joblib import Parallel, delayed 

# ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# print(os.getcwd())
# from __init__.py import ROOT_DIR  # Import the path from __init__.py

# only get ROOT_DIR from __init__.py if I am run as a module 

# try:
#     # Try to import from the package first
#     from . import ROOT_DIR
# except ImportError:
#     # If that fails, set it directly
#     ROOT_DIR = os.environ.get('EEG_DATA_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


ROOT_DIR = '/Users/user/eeg-ds004504/'

freqBands = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 12),
    "Beta": (12, 30),
}

def subPath(sub, derivatives=True):
    print("subPath", sub)
    # print("ROOT_DIR from __init__", ROOT_DIR)
    
    # Strip 'sub-' prefix if it exists
    sub_id = sub.replace('sub-', '') if isinstance(sub, str) and sub.startswith('sub-') else sub
    
    if derivatives:
       path = os.path.join(ROOT_DIR, 'ds004504', 'derivatives', f'sub-{sub_id}', 'eeg', f'sub-{sub_id}_task-eyesclosed_eeg.set')
    else:
       path = os.path.join(ROOT_DIR, 'ds004504', f'sub-{sub_id}', 'eeg', f'sub-{sub_id}_task-eyesclosed_eeg.set')
    
    if not os.path.exists(path):
        raise FileNotFoundError(f'The path was not found for {sub}, path: {path}')
    print(f"Path handed: {path}")
    return path

def participantsInfoPath():
    return os.path.join(ROOT_DIR, 'ds004504', 'participants.tsv')

'''
ProcessSub gets the power density from a subject. It has 3 modes. Generator, sequential or parallel.
Ironically the parallel mode seems to be the slowest by about 15% and the other two are about tied.
'''
def _psd_generator(epochs, compute_psd):
    for i in range(len(epochs)):
        yield compute_psd(epochs[i])

def processSubPSDs(sub, derivatives=True, method='welch', windowLength=3, stepSize=1.5, mode='generator', n_jobs=1):
    raw = mne.io.read_raw_eeglab(subPath(sub, derivatives), preload=True)
    sfreq = raw.info['sfreq']

    start_times = np.arange(0, raw.times[-1] - windowLength, stepSize)
    events = np.array([[int(t * sfreq), 0, 1] for t in start_times])
    
    epochs = mne.Epochs(
        raw, events, event_id=1, tmin=0, tmax=windowLength,
        baseline=None, detrend=1, preload=True, verbose=False
    )

    freqLow = freqBands['Delta'][0]
    freqHigh = freqBands['Beta'][1]

    def compute_psd(epoch):
        return epoch.compute_psd(fmin=freqLow, fmax=freqHigh, method=method, verbose=False)

    if mode == 'generator':
        return _psd_generator(epochs, compute_psd)

    elif mode == 'sequential':
        return [compute_psd(epochs[i]) for i in range(len(epochs))]

    elif mode == 'parallel':
        return Parallel(n_jobs=n_jobs)(
            delayed(compute_psd)(epochs[i]) for i in range(len(epochs))
        )

    else:
        raise ValueError(f"Invalid mode '{mode}'. Choose from 'generator', 'sequential', or 'parallel'.")


'''
This is for processing the subject without getting the psd's. It gets all the epochs for the subject.
'''
def processSub(sub, derivatives=True, windowLength=3, stepSize=1.5):
    print("processSub", sub)
    raw = mne.io.read_raw_eeglab(subPath(sub, derivatives), preload=True)
    sfreq = raw.info['sfreq']

    start_times = np.arange(0, raw.times[-1] - windowLength, stepSize)
    events = np.array([[int(t * sfreq), 0, 1] for t in start_times]) # [sample_index, previous_event_1d, current_event_id], note, 0 -> means we don't have transitions between events,
    
    epochs = mne.Epochs(
        raw, events, event_id=1, tmin=0, tmax=windowLength,
        baseline=None, detrend=1, preload=True, verbose=False
    )
    
    return epochs



if __name__ == '__main__':

    participantsInfo = pd.read_table(participantsInfoPath())


    #Getting all the participaants 
    #note rename A-sub to a better name such as alzSub , also don't know if this part is necessary
    A_sub = participantsInfo[participantsInfo["Group"] == "A"]["participant_id"].tolist()
    C_sub = participantsInfo[participantsInfo["Group"] == "C"]["participant_id"].tolist()
    D_sub = participantsInfo[participantsInfo["Group"] == "F"]["participant_id"].tolist()
    
    start = time.time()
    epochs = processSub(A_sub[0])
    epoch = epochs[0]
   
    print("printing epoch info")
    # print(epochs.to_data_frame().shape())
    print(epoch.info['ch_names'])
    print(type(epochs))
    print(type(epoch))

   
    print("processSub:", time.time()-start)
       
    start = time.time()
    # Generator mode
    for psd in processSubPSDs(A_sub[0], mode='generator'):
        pass
    print("Generator mode: ", time.time()-start)

    start = time.time()
    # Sequential
    psds = processSubPSDs(A_sub[0], mode='sequential')
    print("Sequential mode: ", time.time()-start)

    start = time.time()
    # Parallel
    psds_parallel = processSubPSDs(A_sub[0], mode='parallel', n_jobs=-1)
    print("Parallel mode: ", time.time()-start)



    # subPath(sub=A_sub[0])
    # subPath(sub=A_sub[0],  derivatives=False)
    # gen = processSub(A_sub[0])
    # first_psd = next(gen)
    # print(gen)
    

    '''
    psds = processSub(A_sub[0])
    start = time.time()
    print(psds[0])
    psds[0].plot()
    print(start - time.time())
    '''
    
