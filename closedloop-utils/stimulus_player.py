from __future__ import annotations
import time, threading, queue, math, sys
from typing import Sequence, Literal

import numpy as np
import sounddevice as sd
from pylsl import StreamInfo, StreamOutlet, local_clock
from scipy.signal import lfilter

"""
Contains 1) audio tone / stimulus builders; 2) audio worker class for playback and LSL logging
"""

# ----------------------------------------------------------------------
# AUDIO OUTPUT/TONE VARIANTS ------------------------------------------------

from scipy.signal import butter, sosfilt

def highpass(x, fs, cutoff=200, order=4):
    sos = butter(order, cutoff, btype="highpass", fs=fs, output="sos")
    return sosfilt(sos, x)

def pink_noise(n_samples: int, rms: float = 0.20, rng: np.random.Generator | None = None):
    """Voss–McCartney: add 1/f random octaves → pink noise"""
    rng = rng or np.random.default_rng()
    white = rng.standard_normal(n_samples)
    # Simple 1-pole filter ≈ 1/f
    b, a = [0.049922035, 0.050612699, 0.050612699, 0.049922035], [1, -2.494956, 2.017265, -0.522190]
    pink = lfilter(b, a, white)
    pink = pink / np.std(pink) * rms       # normalise power
    
    # now highpass filter for more stable sound (at least necessary on macbook speakers):
    pink = highpass(pink, fs=48000, cutoff=400, order=4)
    
    return pink.astype(np.float32)

def morlet_wavelet(sample_rate: int, rms: float = 0.05) -> np.ndarray:
    """Fixed Morlet (10 Hz, 6 cycles, auto duration)."""
    FREQ   = 10.0
    CYCLES = 6.0
    duration_s = CYCLES / FREQ
    t = np.arange(-duration_s/2, duration_s/2, 1/sample_rate, dtype=np.float64)
    sigma_t = CYCLES / (2 * np.pi * FREQ)
    envelope = np.exp(-t**2 / (2 * sigma_t**2))
    carrier  = np.cos(2 * np.pi * FREQ * t)
    w = envelope * carrier
    peak = np.max(np.abs(w))
    if peak > 0: w = w / peak
    cur = np.sqrt(np.mean(w**2))
    if cur > 0 and rms > 0: w = w / cur * rms
    return w.astype(np.float32)

def build_stim(
    stim_type: Literal['pink', 'morlet'],
    stim_dur: float,
    fs_out: int,
    stim_rms: float,
    stim_peak: float | None = None,
    stim_peak_normalize: bool = False,
    rng_seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(rng_seed) if rng_seed is not None else None
    if stim_type == 'pink':
        n = int(round(stim_dur * fs_out))
        stim = pink_noise(n, rms=stim_rms, rng=rng)
    else:  # 'morlet' (fixed params)
        stim = morlet_wavelet(fs_out, rms=stim_rms)
    if stim_peak is None:
        return _apply_fade(stim, fs_out, fade_ms=5.0)
    peak = float(np.max(np.abs(stim)))
    if peak <= 0:
        return _apply_fade(stim, fs_out, fade_ms=5.0)
    if stim_peak_normalize or peak > stim_peak:
        stim = (stim / peak) * float(stim_peak)
    return _apply_fade(stim, fs_out, fade_ms=5.0)


def _apply_fade(stim: np.ndarray, fs_out: int, fade_ms: float = 2.0) -> np.ndarray:
    if fade_ms <= 0:
        return stim
    n_fade = int(round((fade_ms / 1000.0) * fs_out))
    n_fade = min(n_fade, len(stim) // 2)
    if n_fade <= 0:
        return stim
    ramp = np.linspace(0.0, 1.0, n_fade, dtype=stim.dtype)
    stim = stim.copy()
    stim[:n_fade] *= ramp
    stim[-n_fade:] *= ramp[::-1]
    return stim


# ----------------------------------------------------------------------
# Audio worker (callback) -------------------------------------------

class AudioWorker:
    """
    Plays short bursts of either 'pink' noise (duration = stim_dur)
    or a fixed Morlet pulse (10 Hz, 6 cycles, auto duration).
    """
    def __init__(self,
                 stim_type: Literal['pink','morlet'] = 'pink',
                 stim_dur: float = 0.1,
                 fs_out: int = 48_000,
                 stim_rms: float = 0.05,
                 stim_peak: float | None = None,
                 stim_peak_normalize: bool = False,
                 rng_seed: int = 42):
        self.fs_out   = int(fs_out)
        self.stim_type = stim_type
        self.stim_dur = float(stim_dur)
        self.stim_rms = float(stim_rms)
        self.stim_peak = stim_peak
        self.stim_peak_normalize = bool(stim_peak_normalize)
        self.rng_seed = rng_seed

        self.stim = self._make_stim()

        self.cmd_q: "queue.Queue[None]" = queue.Queue()

        info = StreamInfo('StimMarkers', 'Markers', 1, 0, 'string', 'stim123')
        self.lsl_out = StreamOutlet(info)

        self._play_buf: np.ndarray | None = None
        self._play_idx = 0

        self.stream = sd.OutputStream(channels=1,
                                      samplerate=self.fs_out,
                                      dtype='float32',
                                      callback=self._callback,
                                      blocksize=0)
        self.stream.start()

    def _make_stim(self) -> np.ndarray:
        return build_stim(
            stim_type=self.stim_type,
            stim_dur=self.stim_dur,
            fs_out=self.fs_out,
            stim_rms=self.stim_rms,
            stim_peak=self.stim_peak,
            stim_peak_normalize=self.stim_peak_normalize,
            rng_seed=self.rng_seed,
        )

    def set_stim_type(self, stim_type: Literal['pink','morlet']):
        if stim_type not in ('pink','morlet'):
            raise ValueError("stim_type must be 'pink' or 'morlet'")
        self.stim_type = stim_type
        self.stim = self._make_stim()

    def trigger(self, ts=None):
        self.cmd_q.put(ts)

    def _callback(self, outdata, frames, t, status):
        if status: print("⚠️", status, file=sys.stderr)

        if self._play_buf is None or self._play_idx >= len(self._play_buf):
            outdata[:] = 0
        else:
            n = min(frames, len(self._play_buf) - self._play_idx)
            outdata[:n, 0] = self._play_buf[self._play_idx:self._play_idx + n]
            if n < frames: outdata[n:, 0] = 0
            self._play_idx += n

        try:
            while True:
                fire_ts = self.cmd_q.get_nowait()
                self._play_buf = self.stim
                self._play_idx = 0
                marker = f'stim_sent:{self.stim_type}'
                if fire_ts is not None:
                    marker += f':det_ts={fire_ts:.4f}'
                stim_ts = local_clock()
                self.lsl_out.push_sample([marker], timestamp=stim_ts)

                # Optional ToDo: push the LSL marker using the PortAudio DAC time (timeinfo.outputBufferDacTime) for better (?) alignment between the first nonzero audio sample and the marker timestamp.

        except queue.Empty:
            pass

    def close(self):
        try:
            self.stream.stop(); self.stream.close()
        except Exception:
            pass

            
