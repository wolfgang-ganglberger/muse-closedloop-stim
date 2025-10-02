from __future__ import annotations
import time, threading, queue, math, sys
from typing import Sequence

import numpy as np
from scipy.signal import butter, lfilter

from osc_receiver import MuseOSCReceiver


# ----------------------------------------------------------------------
# Detector thread -----------------------------------------------------
class DetectorThread(threading.Thread):
    """
    Pulls latest EEG from MuseOSCReceiver, runs causal band-pass,
    fires stimulus when abs(signal) crosses thresh & refractory ok.
    """
    def __init__(self, rx: MuseOSCReceiver, out_q: "queue.Queue[float]",
                 fs: int = 256, band: Sequence[float] = (0.5, 4),
                 thresh_uV: float = 100.0, refractory_s: float = 1.5):
        super().__init__(daemon=True)
        self.rx, self.out_q = rx, out_q
        self.fs, self.thresh = fs, thresh_uV
        self.refractory_s = refractory_s
        self._last_fire = -math.inf

        # causal Butterworth
        b, a = butter(2, np.array(band) / (0.5 * fs), btype='bandpass')
        self._zi = np.zeros(max(len(a), len(b)) - 1)   # filter state
        self._ba = (b, a)

    def run(self):
        ch_slice = slice(0, 1)   # use first EEG channel
        while True:
            eeg = self.rx.get('/eeg', ch_slice)
            if eeg.shape[0] < 2:
                time.sleep(0.05); continue
            t = eeg[-1, 0]
            
            # --- Strict refractory: skip processing if within refractory period ---
            if (t - self._last_fire) < self.refractory_s:
                time.sleep(0.01)
                continue
            
            sig = eeg[:, 1]

            # filter latest chunk
            window_sec = 2
            n_new = min(len(sig), self.fs * window_sec)
            chunk = sig[-n_new:]
            y, self._zi = lfilter(*self._ba, chunk, zi=self._zi)

            detection_window_samples = 5  # scan only last few samples for detection threshold
            if np.max(np.abs(y[-detection_window_samples:])) > self.thresh:
                # time t: detected event, store detection timestamp and add to output queue
                self._last_fire = t
                self.out_q.put(t)
            time.sleep(0.01)
