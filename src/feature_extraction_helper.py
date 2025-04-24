# eeg_feature_extraction.py
import numpy as np
from scipy.stats import skew, kurtosis

# === Basic Stats ===
def compute_mean(x):
    return np.mean(x, axis=1)

def compute_std(x):
    return np.std(x, axis=1)

def compute_variance(x):
    return np.var(x, axis=1)

def compute_rms(x):
    return np.sqrt(np.mean(x**2, axis=1))

# === Higher Order Moments ===
def compute_skewness(x):
    return np.apply_along_axis(skew, axis=1, arr=x)

def compute_kurtosis(x):
    return np.apply_along_axis(kurtosis, axis=1, arr=x)

# === Hjorth Parameters ===
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

# === Entropy ===
def compute_app_entropy(x, m=2, r=0.2):
    from numpy.linalg import norm

    def _phi(signal, m, r):
        N = len(signal)
        x = np.array([signal[i:i + m] for i in range(N - m + 1)])
        C = np.sum([np.sum(norm(x - xi, ord=np.inf, axis=1) <= r) - 1 for xi in x], axis=0)
        return np.log(C / (N - m + 1))

    return np.array([_phi(ch, m, r * np.std(ch)) - _phi(ch, m + 1, r * np.std(ch)) for ch in x])

def compute_samp_entropy(x, m=2, r=0.2):
    def _sampen(U, m, r):
        N = len(U)
        xmi = np.array([U[i:i + m] for i in range(N - m)])
        xmj = np.array([U[i:i + m] for i in range(N - m)])
        xm1i = np.array([U[i:i + m + 1] for i in range(N - m - 1)])
        xm1j = np.array([U[i:i + m + 1] for i in range(N - m - 1)])
        B = np.sum([np.sum(np.max(np.abs(xmi - xmj), axis=1) <= r) - 1 for i in range(len(xmi))])
        A = np.sum([np.sum(np.max(np.abs(xm1i - xm1j), axis=1) <= r) - 1 for i in range(len(xm1i))])
        return -np.log(A / B) if B != 0 and A != 0 else np.nan

    return np.array([_sampen(ch, m, r * np.std(ch)) for ch in x])

# === Fractal Dimensions ===
def compute_higuchi_fd(x, kmax=10):
    def higuchi_fd(ts, kmax):
        N = len(ts)
        Lmk = []
        for k in range(1, kmax+1):
            Lm = []
            for m in range(k):
                L = 0
                n_max = int(np.floor((N - m - 1) / k))
                for i in range(1, n_max):
                    L += abs(ts[m + i * k] - ts[m + (i - 1) * k])
                L = (L * (N - 1)) / (k * n_max * k)
                Lm.append(L)
            Lmk.append(np.mean(Lm))
        logL = np.log(Lmk)
        logk = np.log(1.0 / np.arange(1, kmax + 1))
        return np.polyfit(logk, logL, 1)[0]

    return np.array([higuchi_fd(ch, kmax) for ch in x])

def compute_katz_fd(x):
    def katz_fd(ts):
        L = np.sum(np.abs(np.diff(ts)))
        d = np.max(np.abs(ts - ts[0]))
        n = len(ts)
        return np.log10(n) / (np.log10(n) + np.log10(L / d)) if d != 0 else np.nan

    return np.array([katz_fd(ch) for ch in x])

# === PSD-based Features ===
def spectral_entropy_from_psd(psd_band):
    psd_norm = psd_band / psd_band.sum()
    return -np.sum(psd_norm * np.log2(psd_norm + 1e-12))

def total_band_power(normalPsd, channel_idx=0):
    return normalPsd[channel_idx, :].mean()

