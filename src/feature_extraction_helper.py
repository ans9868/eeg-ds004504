# eeg_feature_extraction.py
import numpy as np
from scipy.stats import skew, kurtosis




def bandPower(normalPsd, freqs, fmin, fmax, channel_idx=0):
   # Select the channel's PSD and the frequencies in the range
    band_mask = (freqs >= fmin) & (freqs < fmax)
    band_power = normalPsd[channel_idx, band_mask].mean()
    
    return band_power


def totalBandPower(normalPsd, freqs, channel_idx=0):    
    total_power = normalPsd[channel_idx, :].mean()
    
    return total_power

def totalEnergy(normalPsd, freqs, channel_idx=0):
    channel_signal = normalPsd[channel_idx, :]
    energy = np.sum(np.square(channel_signal))
    return energy



# * Basic Stats *
def compute_mean(x):
    return np.mean(x, axis=1)

def compute_std(x):
    return np.std(x, axis=1)

def compute_variance(x):
    return np.var(x, axis=1)

def compute_rms(x):
    return np.sqrt(np.mean(x**2, axis=1))


# * Hjorth Parameters *
def compute_hjorth_mobility(x):
    dx = np.diff(x, axis=1)
    return np.sqrt(np.var(dx, axis=1) / np.var(x, axis=1))

def compute_hjorth_complexity(x):
    dx = np.diff(x, axis=1)
    ddx = np.diff(dx, axis=1)
    mobility = compute_hjorth_mobility(x)
    mobility_dx = np.sqrt(np.var(ddx, axis=1) / np.var(dx, axis=1))
    return mobility_dx / mobility

def compute_hjorth_index(x):
    mobility = compute_hjorth_mobility(x)
    complexity = compute_hjorth_complexity(x)
    return 2 * complexity + 100 / (2 * mobility)




# * PSD-based Features *
def spectral_entropy_from_psd(psd_band):
    psd_norm = psd_band / psd_band.sum()
    return -np.sum(psd_norm * np.log2(psd_norm + 1e-12))

def total_band_power(normalPsd, channel_idx=0):
    return normalPsd[channel_idx, :].mean()

