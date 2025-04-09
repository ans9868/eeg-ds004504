import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import mne
import os
import time
from joblib import Parallel, delayed 

try:
    # When run as part of a package (local scripts, Jupyter, etc.)
    from src.config_handler import load_config, initiate_config
    from src.preprocess_sets import processSub, participantsInfoPath
except ImportError:
    # When run inside Spark workers (which get flat files via sc.addPyFile)
    from config_handler import load_config, initiate_config
    from preprocess_sets import processSub, participantsInfoPath


try:
    config = load_config()
except RuntimeError:
    print("Config not found in feature_extraction.py")
    config = initiate_config()


config = load_config()
freqBands = config['freqBands']
windowLength = config['windowLength']
stepSize = config['stepSize']
method = config['method']


def bandPower(normalPsd, freqs, fmin, fmax, channel_idx=0):
   # Select the channel's PSD and the frequencies in the range
    band_mask = (freqs >= fmin) & (freqs < fmax)
    # print(f'Normal psd shape {normalPsd.shape()}')
    #print(f'Normal psd {normalPsd}')
    band_power = normalPsd[channel_idx, band_mask].mean()
    
    return band_power


def totalBandPower(normalPsd, freqs, channel_idx=0):    # But we compute the mean across all frequencies to be consistent
    total_power = normalPsd[channel_idx, :].mean()
    
    return total_power

def processEpoch(epoch, freqBands=freqBands, method=method, windowLength=windowLength, stepSize=stepSize, n_jobs=1):
    """
    Process an epoch to extract band power features for each channel and frequency band.
    
    Parameters:
    -----------
    epoch : mne.Epochs
        The EEG epoch to process.
    freqBands : dict
        Dictionary with band names as keys and (fmin, fmax) tuples as values.
    method : str, optional
        Method to compute PSD (method or 'multitaper').
    windowLength : float, optional
        Length of the window for PSD calculation.
    stepSize : float, optional
        Step size for the window.
    n_jobs : int, optional
        Number of jobs to run in parallel.
        
    Returns:
    --------
    list
        List of tuples: ((channel_name, band_name), feature_value)
    """
    # Determine the overall frequency range
    fmin = min(band_range[0] for band_range in freqBands.values())
    fmax = max(band_range[1] for band_range in freqBands.values())
    
    # Get channel names
    channelNames = epoch.info['ch_names']
    
    # Compute PSD
    psds, freqs = epoch.compute_psd(
        method=method, 
        picks='eeg', 
        fmin=fmin, 
        fmax=fmax, 
        verbose=False
    ).get_data(return_freqs=True)
    
    # Normalize the PSDs (per channel)
    # This makes sure each channel's PSD sums to 1
    normalPsds = psds / np.sum(psds, axis=-1, keepdims=True)
    normalPsds = np.squeeze(normalPsds) 

    # Extract features
    features = []
    
    # Process each channel
    for channel_idx, channel_name in enumerate(channelNames):
        # Calculate each frequency band power
        for band_name, (band_fmin, band_fmax) in freqBands.items():
            band_power = bandPower(normalPsds, freqs, band_fmin, band_fmax, channel_idx)
            features.append(((channel_name, band_name), [band_power])) #will add other stuff next to band power here so that it is iterable
        
            #MAKE IT SO THAT ITERABLE AND EACH CHANNEL NAME / DATAPOINT IS ITERABLE FOR SAME CHANNEL NAME BAND NAME AND BAND POWER !!
        # Calculate total band power
        total_power = totalBandPower(normalPsds, freqs, channel_idx)
        features.append(((channel_name, 'Total'), [total_power])) #is there more stuff that is 'total for the channe, if so add it to the tuble with total power!
    
    return features



'''
Process's a specific subject 
'''
def processSubject(subject, n_jobs=-1, freqBands=freqBands):
    start = time.time()    

    epochs = processSub(subject)
    epochResults = Parallel(n_jobs=n_jobs, prefer="processes")(delayed(processEpoch)(epochs[x], method=method) for x in range(len(epochs)))
    # processedEpoch = processEpoch(epochs[0], freqBands)
   
    print(f"processSubject {subject}:", time.time()-start)
    return epochResults

'''
Put in a list of subjects, and it will process them in a dataframe with numpy arrays for data
'''
def processSubjects(subjectList, n_jobs=-1):
    allResults = {}

    for subject in subjectList:
        result = processSubject(subject, n_jobs=n_jobs)
        allResults[subject] = result

    return allResults


if __name__ == '__main__':
    initiate_config()
    participantsInfo = pd.read_table(participantsInfoPath())
    A_sub = participantsInfo[participantsInfo["Group"] == "A"]["participant_id"].tolist()

    NUM_SUBJECTS = 1
    target_subjects = A_sub[:NUM_SUBJECTS]
    print(target_subjects)
    subject_results = processSubjects(target_subjects)  # limit to 2 for quick testing

    for subject, df in subject_results.items():
        print(f"\nSubject: {subject}")
        print(df)


    '''
    participantsInfo = pd.read_table('./ds004504/participants.tsv')
    A_sub = participantsInfo[participantsInfo["Group"] == "A"]["participant_id"].tolist()
    C_sub = participantsInfo[participantsInfo["Group"] == "C"]["participant_id"].tolist()
    D_sub = participantsInfo[participantsInfo["Group"] == "F"]["participant_id"].tolist()
   
    start = time.time()
    epochs = processSub(A_sub[0])
    processedEpoch = processEpoch(epochs[0], freqBands)
    print("processSub:", time.time()-start)
    
    print(processedEpoch)
    start = time.time()
    '''
