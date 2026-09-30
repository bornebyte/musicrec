"""Audio -> 81-dim feature vector. No beat tracker (numba-heavy, crashed on py3.14); tempo comes from
tempogram autocorrelation instead.

Blocks (equalised later so no block dominates by size):
  timbre  : MFCC mean/std (20+20) + spectral contrast (7)          = 47
  harmony : chroma mean/std (12+12)                                = 24
  rhythm  : tempo, onset mean, onset std, periodicity strength     = 4
  energy  : rms mean/std, centroid, bandwidth, rolloff, zcr        = 6
"""
import numpy as np
import librosa

from . import config as C
from .audio import decode

BLOCK_SIZES = {"timbre": 47, "harmony": 24, "rhythm": 4, "energy": 6}
BLOCK_ORDER = ["timbre", "harmony", "rhythm", "energy"]
assert sum(BLOCK_SIZES.values()) == C.FEATURE_DIM

HOP = 512


def block_slices():
    out, i = {}, 0
    for b in BLOCK_ORDER:
        out[b] = slice(i, i + BLOCK_SIZES[b])
        i += BLOCK_SIZES[b]
    return out


def _tempo(onset_env, sr):
    try:
        from librosa.feature.rhythm import tempo as fn
    except ImportError:
        from librosa.beat import tempo as fn
    return float(np.atleast_1d(fn(onset_envelope=onset_env, sr=sr, hop_length=HOP))[0])


def extract_features(path, duration: float) -> np.ndarray:
    off = max(0.0, duration / 2 - C.CLIP_SECONDS / 2)
    y = decode(path, off, C.CLIP_SECONDS, C.SR)
    sr = C.SR
    if y.size < sr * 3:
        raise ValueError("audio shorter than 3 s")

    y_h, y_p = librosa.effects.hpss(y)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    timbre = np.concatenate([mfcc.mean(1), mfcc.std(1), contrast.mean(1)])

    chroma = librosa.feature.chroma_stft(y=y_h, sr=sr)
    harmony = np.concatenate([chroma.mean(1), chroma.std(1)])

    onset = librosa.onset.onset_strength(y=y_p, sr=sr, hop_length=HOP)
    tempo = _tempo(onset, sr)
    ac = librosa.autocorrelate(onset - onset.mean(), max_size=int(sr / HOP * 1.2))
    lo, hi = int(sr / HOP * 60 / 200), int(sr / HOP * 60 / 60)      # 60-200 BPM lag window
    periodicity = float(ac[lo:hi].max() / (ac[0] + 1e-9)) if ac[0] > 0 and hi > lo else 0.0
    rhythm = np.array([tempo, onset.mean(), onset.std(), periodicity])

    rms = librosa.feature.rms(y=y)[0]
    energy = np.array([
        rms.mean(), rms.std(),
        librosa.feature.spectral_centroid(y=y, sr=sr).mean(),
        librosa.feature.spectral_bandwidth(y=y, sr=sr).mean(),
        librosa.feature.spectral_rolloff(y=y, sr=sr).mean(),
        librosa.feature.zero_crossing_rate(y).mean(),
    ])

    v = np.concatenate([timbre, harmony, rhythm, energy]).astype(np.float32)
    assert v.size == C.FEATURE_DIM
    return np.nan_to_num(v)
