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
except ImportError:
    # When run inside Spark workers (which get flat files via sc.addPyFile)
    from preprocess_sets import processSub, participantsInfoPath
    from config_handler import load_config, initiate_config


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


# TODO : more feature extractoin per electrode per band  , be smark about what choose , and make it so that do it efficiently / no repeats 
# At least make it so taht we expand the electrode level features, then maybe are able to do the band level features depending on time factor 
def processEpoch(subjectID, epochID, epoch, freqBands=freqBands, method=method, windowLength=windowLength, stepSize=stepSize, n_jobs=1):
    fmin = min(band_range[0] for band_range in freqBands.values())
    fmax = max(band_range[1] for band_range in freqBands.values())
    
    channelNames = epoch.info['ch_names']
    
    psds, freqs = epoch.compute_psd(
        method=method,
        picks='eeg',
        fmin=fmin,
        fmax=fmax,
        verbose=False
    ).get_data(return_freqs=True)
    
    normalPsds = psds / np.sum(psds, axis=-1, keepdims=True)
    normalPsds = np.squeeze(normalPsds)

    rows = []
    data = epoch.get_data(picks="eeg")[0]  # shape (n_channels, n_times)
    
    for channel_idx, channel_name in enumerate(channelNames):
        for band_name, (band_fmin, band_fmax) in freqBands.items():
            band_power = bandPower(normalPsds, freqs, band_fmin, band_fmax, channel_idx)
            rows.append(Row(
                SubjectID=subjectID,
                EpochID=epochID,
                Electrode=channel_name,
                WaveBand=band_name,
                FeatureName="Power",
                FeatureValue=band_power,
                table_type="band"
            ))

        rows.append(Row(
            SubjectID=subjectID,
            EpochID=epochID,
            Electrode=channel_name,
            WaveBand=None,
            FeatureName="TotalEnergy",
            FeatureValue=totalEnergy(normalPsds, freqs, channel_idx),
            table_type="electrode"
        ))

        rows.append(Row(
            SubjectID=subjectID,
            EpochID=epochID,
            Electrode=channel_name,
            WaveBand=None,
            FeatureName="TotalPower",
            FeatureValue=totalBandPower(normalPsds, freqs, channel_idx),
            table_type="electrode"
        ))




    # ----- Epoch-level features via mne-features -----
    data = epoch.get_data(picks="eeg")[0]  # shape (n_channels, n_times)

    # univariate features - averaged aross channels
    add_epoch_feature(rows, subjectID, epochID, "Mean", np.mean(compute_mean(data)))
    add_epoch_feature(rows, subjectID, epochID, "Std", np.mean(compute_std(data)))
    add_epoch_feature(rows, subjectID, epochID, "Variance", np.mean(compute_std(data) ** 2))
    add_epoch_feature(rows, subjectID, epochID, "Skewness", np.mean(compute_skewness(data)))
    add_epoch_feature(rows, subjectID, epochID, "Kurtosis", np.mean(compute_kurtosis(data)))
    add_epoch_feature(rows, subjectID, epochID, "RMS", np.mean(compute_rms(data)))
    
    # Hjorth parameters
    add_epoch_feature(rows, subjectID, epochID, "HjorthMobility", np.mean(compute_hjorth_mobility(data)))
    add_epoch_feature(rows, subjectID, epochID, "HjorthComplexity", np.mean(compute_hjorth_complexity(data)))

    # Entropy + nonlinear
    add_epoch_feature(rows, subjectID, epochID, "AppEntropy", np.mean(compute_app_entropy(data)))
    add_epoch_feature(rows, subjectID, epochID, "SampleEntropy", np.mean(compute_samp_entropy(data)))
    add_epoch_feature(rows, subjectID, epochID, "HiguchiFD", np.mean(compute_higuchi_fd(data)))
    add_epoch_feature(rows, subjectID, epochID, "KatzFD", np.mean(compute_katz_fd(data)))

    return rows


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
