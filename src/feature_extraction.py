import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import mne
import os
import time
from joblib import Parallel, delayed 
from preprocess_sets import processSub, participantsInfoPath

freqBands = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 12),
    "Beta": (12, 30),
}


#Make a dicitonary of functions or ... how should we do each funciotn for feature extraction 
def allBandPower(epoch, freqBands, method='welch', windowLength=3, stepSize=1.5, n_jobs=-1):
    fmin=freqBands['Delta'][0] 
    fmax=freqBands['Beta'][1]
    
    channelNames = epoch.info['ch_names']
    
    psds, freqs = epoch.compute_psd(method=method, picks='eeg', fmin=fmin, fmax=fmax, verbose=False).get_data(return_freqs=True)

    #normalize the psd's
    psds /= np.sum(psds, axis=-1, keepdims=True)

    # the band power of all the channels 
    bandPowers = []
    for band_idx, (band_name, (fmin, fmax)) in enumerate(freqBands.items()):
        #the band power of a specific channel
        bandPower = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
        bandPower = np.squeeze(bandPower)

        for channel_idx, channel_name in enumerate(channelNames):
            bandPowers.append(((channel_name, band_name), (bandPower[channel_idx]))) 
            # TODO: i put bandpower into a tupble  but this is a shortcut for now. We need to put all features into a tuple, not sure the best way to do that but it probably involves doing the PSD calculation in the main function
            # *note* if do hte calculation in the main funciotn should try the 2 different methods
    # this should be a dataframe with numpy array funciotns
    # Channel {} band {} window power  [...]
    return bandPowers  # Returns list of tuples: (channel, band) → PSD value


def processEpoch(epoch, freqBands=freqBands, method='welch', windowLength=3, stepSize=1.5, n_jobs=1):
    epochBandPower = allBandPower(epoch, freqBands, method='welch', windowLength=3, stepSize=1.5) #power all 19 channels for Delta Theta ALphba Beta and * Total power * 
    # epochKurtosis = kurtosis(...)
    # epochEntropy = entropy(...)
    # ... 

    # need to combine all the dataframes from each data point here 
    
    dataCombined = [epochBandPower]

    return dataCombined

'''
Process's a specific subject 
'''
def processSubject(subject, n_jobs=-1, freqBands=freqBands):
    start = time.time()    

    epochs = processSub(subject)
    epochResults = Parallel(n_jobs=n_jobs, prefer="processes")(delayed(processEpoch)(epochs[x], method='welch') for x in range(len(epochs)))
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
