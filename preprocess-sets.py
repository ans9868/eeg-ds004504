import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import mne
import os
import time

participantsInfo = pd.read_table('./ds004504/participants.tsv')


#Getting all the participaants 
#note rename A-sub to a better name such as alzSub , also don't know if this part is necessary
A_sub = participantsInfo[participantsInfo["Group"] == "A"]["participant_id"].tolist()
C_sub = participantsInfo[participantsInfo["Group"] == "C"]["participant_id"].tolist()
D_sub = participantsInfo[participantsInfo["Group"] == "F"]["participant_id"].tolist()

freqBands = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 12),
    "Beta": (12, 30),
}

#*******
def subPath(sub, derivatives=True):
    if(derivatives):
       path = f"ds004504/derivatives/{sub}/eeg/{sub}_task-eyesclosed_eeg.set"
    else:
       path = f"ds004504/{sub}/eeg/{sub}_task-eyesclosed_eeg.set"
    
    if not os.path.exists(path):
        raise FileNotFoundError(f'The path was not found for {sub}')

    return path

#I think menthod not here ,what is preload ? 
def processSub(sub, derivatives=True, method='welch', windowLength=3, stepSize=1.5):
    rawSub = mne.io.read_raw_eeglab(subPath(sub, derivatives), preload=True)
    stdFreq = rawSub.info['sfreq'] #standard frequency, it should be 500hz

    startTimes = np.arange(0, rawSub.times[-1] - windowLength, stepSize)

    events = np.array([
    [int(t * stdFreq), 0, 1] for t in startTimes  # MNE event format
    ]) 

    # * we may need to remove gaps from if derivatives is true
    epochs = mne.Epochs(rawSub, events, event_id=1, tmin=0, tmax=windowLength, baseline=None, detrend=1, preload=True, verbose=False)

    freqLow = freqBands['Delta'][0]
    freqHigh = freqBands['Beta'][1]

    psds = []

    # * can multiprocess this part if need be or make it a generator (with yeild)
    for idx in range(len(epochs)):
        epoch = epochs[idx]
        # epochPsd = epoch.compute_psd(fmin=freqLow, fmax=freqHigh, verbose=False)
        epochPsd = epoch.compute_psd(fmin=freqLow, fmax=freqHigh, method=method, verbose=False)
        yield epochPsd 
   
        # psds.append(epochPsd)
    
    # return psds

    # computePsd = epoch.compute_psd(fmin=freqLow, fmax=freqHigh, method=method)
    

if __name__ == '__main__':
    subPath(sub=A_sub[0])
    subPath(sub=A_sub[0],  derivatives=False)
    gen = processSub(A_sub[0])
    first_psd = next(gen)
    print(gen)
    

    '''
    psds = processSub(A_sub[0])
    start = time.time()
    print(psds[0])
    psds[0].plot()
    print(start - time.time())
    '''
    
