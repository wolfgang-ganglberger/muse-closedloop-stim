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

def pink_noise(n_samples: int, rms: float = 0.05, rng: np.random.Generator | None = None):
    """Voss–McCartney: add 1/f random octaves → pink noise"""
    rng = rng or np.random.default_rng()
    white = rng.standard_normal(n_samples)
    # Simple 1-pole filter ≈ 1/f
    b, a = [0.049922035, 0.050612699, 0.050612699, 0.049922035], [1, -2.494956, 2.017265, -0.522190]
    pink = lfilter(b, a, white)
    pink = pink / np.std(pink) * rms       # normalise power
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
                 stim_rms: float = 0.05):
        self.fs_out   = int(fs_out)
        self.stim_type = stim_type
        self.stim_dur = float(stim_dur)
        self.stim_rms = float(stim_rms)

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
        if self.stim_type == 'pink':
            n = int(round(self.stim_dur * self.fs_out))
            return pink_noise(n, rms=self.stim_rms)
        else:  # 'morlet' (fixed params)
            return morlet_wavelet(self.fs_out, rms=self.stim_rms)

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

            