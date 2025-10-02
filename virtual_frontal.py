"""Utilities for synthesizing a virtual frontal EEG channel from Muse temporals."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

__all__ = [
    "make_virtual_frontal_from_temporals",
    "lowpass_delta_frontal",
]


def _butter_design(fs: float, btype: str, cutoff, order: int):
    nyquist = 0.5 * fs
    if btype in {"high", "low"}:
        wn = float(cutoff) / nyquist
    elif btype == "band":
        wn = [c / nyquist for c in cutoff]
    else:
        raise ValueError("btype must be 'high', 'low', or 'band'")
    return butter(order, wn, btype=btype)


def lp_filter(x: np.ndarray, fs: float, fc: float = 3.5, order: int = 3) -> np.ndarray:
    b, a = _butter_design(fs, "low", fc, order)
    return filtfilt(b, a, x)


def bandpass(x: np.ndarray, fs: float, f_lo: float, f_hi: float, order: int = 3) -> np.ndarray:
    b, a = _butter_design(fs, "band", (f_lo, f_hi), order)
    return filtfilt(b, a, x)


def noise_to_weight(noise_vec: np.ndarray, floor: float = 0.0, ceil: float = 1.0) -> np.ndarray:
    weights = 1.0 - np.clip(noise_vec.astype(float), 0.0, 1.0)
    return np.clip(weights, floor, ceil)


def weighted_mean(values: np.ndarray, weights: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    num = np.nansum(values * weights, axis=0)
    den = np.nansum(weights, axis=0)
    result = np.full_like(num, np.nan)
    mask = den > eps
    result[mask] = num[mask] / den[mask]
    return result


def make_virtual_frontal_from_temporals(
    df_eeg: pd.DataFrame,
    fs: float = 256.0,
    max_noise: float = 0.6,
    return_details: bool = False,
):
    """Return a noise-weighted frontal estimate built from Muse temporals.

    Parameters
    ----------
    df_eeg : pandas.DataFrame
        Requires columns 'ch1', 'ch4', 'noise_level_channel_1', 'noise_level_channel_4'.
    fs : float, optional
        Sampling frequency in hertz.
    max_noise : float, optional
        Samples noisier than ``max_noise`` are ignored per channel.
    return_details : bool, optional
        When True return ``(v_frontal, v1, v2, mask_good)`` instead of just ``v_frontal``.

    Returns
    -------
    numpy.ndarray or tuple
        ``v_frontal`` by default, or ``(v_frontal, v1, v2, mask_good)`` when
        ``return_details`` is True.
    """

    tp9 = df_eeg["ch1"].to_numpy()
    tp10 = df_eeg["ch4"].to_numpy()

    v1 = -tp9
    v2 = -tp10

    v1 = bandpass(v1, fs=fs, f_lo=0.1, f_hi=20.0, order=3)
    v2 = bandpass(v2, fs=fs, f_lo=0.1, f_hi=20.0, order=3)

    w1 = noise_to_weight(df_eeg["noise_level_channel_1"].to_numpy())
    w2 = noise_to_weight(df_eeg["noise_level_channel_4"].to_numpy())

    ok1 = w1 >= (1.0 - max_noise)
    ok2 = w2 >= (1.0 - max_noise)

    values = np.vstack([v1, v2])
    weights = np.vstack([
        np.where(ok1, w1, 0.0),
        np.where(ok2, w2, 0.0),
    ])
    v_frontal = weighted_mean(values, weights)
    v_frontal = pd.Series(v_frontal).ffill().bfill().to_numpy()

    mask_good = ok1 | ok2

    if return_details:
        return v_frontal, v1, v2, mask_good
    return v_frontal


def lowpass_delta_frontal(
    v_frontal: np.ndarray,
    fs: float = 256.0,
    fc: float = 3.5,
    order: int = 3,
) -> np.ndarray:
    """Create a low-pass delta-filtered virtual frontal signal."""

    v_frontal = np.asarray(v_frontal, dtype=float)
    return lp_filter(v_frontal, fs=fs, fc=fc, order=order)
