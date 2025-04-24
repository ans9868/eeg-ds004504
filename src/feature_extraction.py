import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import mne
import os
import time
from joblib import Parallel, delayed 
from pyspark.sql import Row
from mne_features.univariate import (
    compute_app_entropy,
    compute_samp_entropy,
    compute_higuchi_fd,
    compute_katz_fd,
    compute_hjorth_mobility,
    compute_hjorth_complexity,
    compute_rms,
    compute_skewness,
    compute_kurtosis,
    compute_std,
    compute_mean
)
try:
    # When run as part of a package (local scripts, Jupyter, etc.)
    from src.config_handler import load_config, initiate_config
    from src.preprocess_sets import processSub, participantsInfoPath
    from src.feature_extraction_helper import *
except ImportError:
    # When run inside Spark workers (which get flat files via sc.addPyFile)
    from preprocess_sets import processSub, participantsInfoPath
    from config_handler import load_config, initiate_config
    from feature_extraction_helper import *

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
print(f"Config using in feature Extraction.py {config}")

def bandPower(normalPsd, freqs, fmin, fmax, channel_idx=0):
   # Select the channel's PSD and the frequencies in the range
    band_mask = (freqs >= fmin) & (freqs < fmax)
    # print(f'Normal psd shape {normalPsd.shape()}')
    #print(f'Normal psd {normalPsd}')
    band_power = normalPsd[channel_idx, band_mask].mean()
    
    return band_power


# check, this might just be all zeros after normliziatoin 
def totalBandPower(normalPsd, freqs, channel_idx=0):    # But we compute the mean across all frequencies to be consistent
    total_power = normalPsd[channel_idx, :].mean()
    
    return total_power

def totalEnergy(normalPsd, freqs, channel_idx=0):
    '''
    Compute the total energy of the EEG signal for a specific channel.
    Total energy is defined as the sum of the squared amplitude over time.     
    '''
   
    channel_signal = normalPsd[channel_idx, :]
    energy = np.sum(np.square(channel_signal))
    return energy

def add_epoch_feature(rows, subjectID, epochID, feature_name, value):
    rows.append(Row(
        SubjectID=subjectID,
        EpochID=epochID,
        Electrode=None,
        WaveBand=None,
        FeatureName=feature_name,
        FeatureValue=float(value),
        table_type="epoch"
    ))



method = 'welch'
windowLength = 3
stepSize = 1.5
from mne.filter import filter_data

from pyspark.sql import Row
def processEpoch(subjectID, epochID, epoch, freqBands=freqBands, method=method, windowLength=windowLength, stepSize=stepSize, n_jobs=1):
    fmin = min(band_range[0] for band_range in freqBands.values())
    fmax = max(band_range[1] for band_range in freqBands.values())

    channelNames = epoch.info['ch_names']

    # PSD-based features
    psds, freqs = epoch.compute_psd(
        method=method,
        picks='eeg',
        fmin=fmin,
        fmax=fmax,
        verbose=False
    ).get_data(return_freqs=True)
    
    normalPsds = psds / np.sum(psds, axis=-1, keepdims=True)
    normalPsds = np.squeeze(normalPsds)

    # Time-domain EEG data for this epoch
    data = epoch.get_data(picks="eeg")[0]  # shape (n_channels, n_times)

    rows = []

    for channel_idx, channel_name in enumerate(channelNames):
        # Electrode-level features (non-band-specific)
        electrode_features = [
            ("TotalPower", totalBandPower(normalPsds, freqs, channel_idx)),
            ("TotalEnergy", totalEnergy(normalPsds, freqs, channel_idx)),
            ("SpectralEntropy", spectral_entropy_from_psd(normalPsds[channel_idx, :])),
            ("HjorthActivity", np.var(data[channel_idx, :])),
            ("HjorthMobility", compute_hjorth_mobility(data[channel_idx:channel_idx+1])[0]),
            ("HjorthComplexity", compute_hjorth_complexity(data[channel_idx:channel_idx+1])[0]),
            ("HjorthIndex", compute_hjorth_index(data[channel_idx:channel_idx+1])[0])
        ]
        
        for fname, val in electrode_features:
            rows.append(Row(
                SubjectID=subjectID,
                EpochID=epochID,
                Electrode=channel_name,
                WaveBand=None,
                FeatureName=fname,
                FeatureValue=float(val),
                table_type="electrode"
            ))


        for band_name, (band_fmin, band_fmax) in freqBands.items():
            band_mask = (freqs >= band_fmin) & (freqs < band_fmax)
            psd_band = normalPsds[channel_idx, band_mask]

            # Band Power from PSD
            band_power = psd_band.mean()
            spectral_entropy = spectral_entropy_from_psd(psd_band)

            # Optional: Filtered band-passed time-series (replace with real filtered data if available)
            # Here we slice based on band_mask for pseudo-time-domain view (not valid!)
            band_signal = data[channel_idx, :]  # Ideally, you'd band-pass filter this

            mobility = compute_hjorth_mobility(data[channel_idx:channel_idx+1])[0]
            complexity = compute_hjorth_complexity(data[channel_idx:channel_idx+1])[0]
            hjorth_index = compute_hjorth_index(data[channel_idx:channel_idx+1])[0]

            # Band-level features
            for fname, val in [
                ("Power", band_power),
                ("SpectralEntropy", spectral_entropy),
                # ("HjorthActivity", activit),
                ("HjorthMobility", mobility),
                ("HjorthComplexity", complexity),
                ("HjorthIndex", hjorth_index)
            ]:
                rows.append(Row(
                    SubjectID=subjectID,
                    EpochID=epochID,
                    Electrode=channel_name,
                    WaveBand=band_name,
                    FeatureName=fname,
                    FeatureValue=float(val),
                    table_type="band"
                ))    


    # Epoch-level features: averaged across all channels
    epoch_feature_list = [
        ("Mean", np.mean(compute_mean(data))),
        ("Std", np.mean(compute_std(data))),
        ("Variance", np.mean(compute_variance(data))),
        ("Skewness", np.mean(compute_skewness(data))),
        ("Kurtosis", np.mean(compute_kurtosis(data))),
        ("RMS", np.mean(compute_rms(data))),
        ("HjorthMobility", np.mean(compute_hjorth_mobility(data))),
        ("HjorthComplexity", np.mean(compute_hjorth_complexity(data))),
        ("HjorthIndex", np.mean(compute_hjorth_index(data))),
        ("AppEntropy", np.mean(compute_app_entropy(data))),
        ("SampleEntropy", np.mean(compute_samp_entropy(data))),
        ("HiguchiFD", np.mean(compute_higuchi_fd(data))),
        ("KatzFD", np.mean(compute_katz_fd(data)))
    ]
    
    for fname, val in epoch_feature_list:
        rows.append(Row(
            SubjectID=subjectID,
            EpochID=epochID,
            Electrode=None,
            WaveBand=None,
            FeatureName=fname,
            FeatureValue=float(val),
            table_type="epoch"
        ))
        
    return rows



'''
Process's a specific subject , move this to populate schemas ? 
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
