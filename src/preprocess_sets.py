import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import mne
import os
import time
from joblib import Parallel, delayed 

try:
    # When run as part of a package (local scripts, Jupyter, etc.)
    from src.config_handler import initiate_config, load_config
except ImportError:
    # When run inside Spark workers (which get flat files via sc.addPyFile)
    from config_handler import initiate_config, load_config


try:
    config = load_config()
except RuntimeError:
    print("Config not found in feature_extraction.py")
    config = initiate_config()


config = load_config()
data_path = config['data_path']
derivatives = config['derivatives']
freqBands = config['freqBands']
windowLength = config['windowLength']
stepSize = config['stepSize']
method = config['stepSize']


def get_data_path():
    return data_path

def subPath(sub, derivatives=True):
    print("subPath", sub)
    print("subPath data_path", data_path)
    
    sub_id = sub.replace('sub-', '') if isinstance(sub, str) and sub.startswith('sub-') else sub
    print(f"Derivatives: {derivatives}") 
    print(f"Derivatives from config: {config['derivatives']}") 

    if derivatives:
       path = os.path.join(data_path, 'ds004504', 'derivatives', f'sub-{sub_id}', 'eeg', f'sub-{sub_id}_task-eyesclosed_eeg.set')
    else:
       path = os.path.join(data_path, 'ds004504', f'sub-{sub_id}', 'eeg', f'sub-{sub_id}_task-eyesclosed_eeg.set')
    
    if not os.path.exists(path):
        raise FileNotFoundError(f'The path was not found for {sub}, path: {path}')
    print(f"Path handed: {path}")
    return path

def participantsInfoPath():
    return os.path.join(data_path, 'ds004504', 'participants.tsv')

'''
ProcessSub gets the power density from a subject. It has 3 modes. Generator, sequential or parallel.
Ironically the parallel mode seems to be the slowest by about 15% and the other two are about tied.
'''
def _psd_generator(epochs, compute_psd):
    for i in range(len(epochs)):
        yield compute_psd(epochs[i])

def processSubPSDs(sub, derivatives=True, method=method, windowLength=windowLength, stepSize=stepSize, mode='generator', n_jobs=1):
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
def processSub(sub, derivatives=derivatives, windowLength=windowLength, stepSize=stepSize):
    print("processSub", sub)
    print("processSub: derivatives", derivatives)
    print("processSub: windowLength", windowLength)
    print("processSub: windowLength", stepSize)
    raw = mne.io.read_raw_eeglab(subPath(sub, derivatives), preload=True)
    
    # removing the boundary events by adding BAD_boundary
    for i, desc in enumerate(raw.annotations.description):
        if 'boundary' in desc:
            raw.annotations.description[i] = 'BAD_boundary'

    sfreq = raw.info['sfreq']


    start_times = np.arange(0, raw.times[-1] - windowLength, stepSize)
    events = np.array([[int(t * sfreq), 0, 1] for t in start_times]) # [sample_index, previous_event_1d, current_event_id], note, 0 -> means we don't have transitions between events,
    
    epochs = mne.Epochs(
        raw, events, event_id=1, tmin=0, tmax=windowLength,
        baseline=None, detrend=1, preload=True, verbose=False,
        reject_by_annotation=True
    )
    
    return epochs
