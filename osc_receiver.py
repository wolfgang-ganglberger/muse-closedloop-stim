#!/usr/bin/env python3
"""
====================
Light-weight liblo ServerThread that buffers Muse OSC packets
and makes them accessible as NumPy arrays in real time.
"""

import logging
import time
import signal
from collections import deque, defaultdict
from threading import Lock
from typing import Dict, Deque, Tuple, Any, Sequence, Mapping

import numpy as np
import pyliblo3 as liblo

import subprocess

from time_regularization import regularize_timestamps 

use_lsl_clock = True
if use_lsl_clock:
    from pylsl import local_clock


# -----------------------------------------------------------------------------
# Ensure the UDP port is free before starting the receiver
# ------------------------------------------------------------------------------
def free_osc_port(port):
    """
    Free the specified UDP port by killing any processes using it.
    This is a safety measure to ensure the port is available for the MuseOSCReceiver.
    """
    # Find processes using the port
    try:
        result = subprocess.run(['lsof', '-i', f':{port}'],
                                capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        if len(lines) > 1:  # Header + at least one process
            print(f"Port {port} is in use. Attempting to kill the process.")
            for line in lines[1:]:
                parts = line.split()
                pid = parts[1]  # Second column is PID
                print(f"Killing PID: {pid}")
                subprocess.run(['kill', '-9', pid])
        else:
            print(f"Port {port} is free.")
    except subprocess.CalledProcessError:
        print(f"Port {port} is free.")


# -----------------------------------------------------------------------------
# Timestamp generator/fixer for synthetic clocks
# -----------------------------------------------------------------------------
class PLLClock:
    """
    Keeps a synthetic clock at sampling-rate fs, nudged toward host timestamps
    with a first-order PLL.  Works for single samples *or* bursts of length n.

    Parameters
    ----------
    fs : float
        Target sampling rate in Hz.
    alpha : float, 0<α<=1
        Pull-in coefficient. Smaller α = slower, smoother correction.
    max_gap : float
        If host clock jumps by > max_gap seconds, re-sync immediately.
    """

    def __init__(self, fs: float, alpha: float = 0.02, max_gap: float = 1.0):
        self.fs      = fs
        self.dt      = 1.0 / fs
        self.alpha   = alpha
        self.max_gap = max_gap
        self._t      = None      # internal running time

    # ---------------------------------------------------------------------
    def assign(self, host_ts: float, n: int = 1):
        """
        Given the host arrival-time of *n* consecutive samples, return an
        array of n uniformly-spaced synthetic timestamps.

        host_ts : float
            Wall-clock timestamp of the most-recent sample in the burst
            (same domain as the rest of your pipeline; use pylsl.local_clock() if you want LSL time end-to-end).
        n : int
            How many samples are in this burst (== len(args)/n_chan).
        """
        if self._t is None:
            # first packet → lock so last sample aligns to host
            self._t = host_ts
        else:
            drift = host_ts - self._t    # how far behind/ahead are we?
            if abs(drift) > self.max_gap:
                # huge gap (e.g. link dropped) → snap
                self._t = host_ts
            else:
                # nudge clock toward host time by alpha*drift
                self._t += self.alpha * drift

        # produce timestamps for the burst
        t0   = self._t - (n - 1) * self.dt   # first sample in burst
        ts   = t0 + np.arange(n) * self.dt

        # advance internal clock for next call
        self._t = ts[-1] + self.dt
        return ts
    
    

STREAM_META = {
    "/eeg":  dict(n_chan=6, fs=256),
    "/ppg":  dict(n_chan=3, fs=64),
    "/acc":  dict(n_chan=3, fs=52),
    "/muse_metrics": dict(n_chan=30, fs=64),
    "/headband_on": dict(n_chan=1, fs=64),
    "/hsi":  dict(n_chan=4, fs=64),
    "/is_good": dict(n_chan=1, fs=64),
}


class MuseOSCReceiver(liblo.ServerThread):
    """
    Start with:
        rx = MuseOSCReceiver(port=7000)
        rx.start()          # returns immediately; runs in its own thread
    Grab data anywhere:
        eeg = rx.get('/eeg')        # shape (n, 6)  -> [timestamp, ch0, ch1, ch2, ch3, ch4, ch5]
        acc = rx.get('/acc')        # shape (n, 4)  -> [timestamp, x, y, z]
    """

    DEFAULT_PATHS = ("/eeg", "/ppg", "/acc", "/muse_metrics",
                     "/headband_on", "/hsi", "/is_good")

    def __init__(self, port: int = 7000, buf_seconds: int = 30,
                 samp_rates: Dict[str, int] | None = None) -> None:
        """
        Parameters
        ----------
        port : int
            UDP port to listen on (matches Muse app output).
        buf_seconds : int
            Length of circular buffer to keep per path.
        samp_rates : dict
            Optional mapping of OSC path → expected sample rate.
            Used to size the deques; falls back to 256 Hz.
        """
        super().__init__(port)
        self._lock: Lock = Lock()

        # Pre-allocate reasonable ring-buffers for each stream
        self._buf: Dict[str, Deque[Tuple[float, Sequence[Any]]]] = {}
        for p in self.DEFAULT_PATHS:
            sr = (samp_rates or {}).get(p, 256)
            maxlen = sr * buf_seconds
            self._buf[p] = deque(maxlen=maxlen)

        # catch-all handler
        self.add_method(None, None, self._handler)
        
        # Set up a PLL clock for synthetic timestamps
        self._pll = {
            "/eeg":  PLLClock(256, alpha=0.02),
            "/ppg":  PLLClock(64,  alpha=0.02),
            "/acc":  PLLClock(52,  alpha=0.02),
            "/muse_metrics":  PLLClock(60,  alpha=0.02),
            "/headband_on":  PLLClock(156,  alpha=0.02),
            "/hsi":  PLLClock(156,  alpha=0.02),
            "/is_good":  PLLClock(156,  alpha=0.02),
        }
        
        
    def _handler(self, path, args, types, src) -> None:
        if use_lsl_clock:
            host_now = local_clock()  # LSL time base (seconds)
        else:
            host_now = time.perf_counter()  # perf_counter time base (seconds)
            
        meta = STREAM_META.get(path)
        if meta is None: return
        n_chan = meta["n_chan"]
        fs     = meta["fs"]
        
        pll   = self._pll[path]
        pkt = np.asarray(args, dtype=float)
        
        if pkt.size % n_chan: return
        pkt = pkt.reshape(-1, n_chan)
        n = pkt.shape[0]

        ts_vec = pll.assign(host_now, n) # vectorised timestamps

        with self._lock:
            dq = self._buf.setdefault(path, deque(maxlen=fs*30))
            # print("Burst n:", n, "First t:", ts_vec[0], "Last t:", ts_vec[-1], "dts:", np.diff(ts_vec))
            dq.extend(zip(ts_vec, pkt))
            
                
    # ---------- public API ----------
    def get(self, path: str, ch_slice=slice(None)) -> np.ndarray:
        """
        Return buffered data for `path` as an (n, m+1) array
        where column 0 = timestamps in the selected host clock domain (pylsl.local_clock() seconds if use_lsl_clock=True) and remaining columns = values.
        If no data, returns empty (0, 0) array.
        """
        with self._lock:
            buf = list(self._buf.get(path, []))
        if not buf:
            return np.empty((0, 0))
        t, x = zip(*buf)
        data = np.column_stack([t, np.vstack(x)])
        return data[:, np.r_[0, 1+np.arange(data.shape[1]-1)[ch_slice]]]

    # ---------- convenience ----------
    def stop_gracefully(self) -> None:
        # Wake the server thread with a dummy packet
        try:
            port = getattr(self, "port", 7000)
            liblo.send(liblo.Address(port), "/shutdown", 1)
        except Exception as e:
            # print(f"Warning: {e}")  # Ignore if server already stopped
            pass  # Safe to ignore (if server already stopped, etc.)
        self.stop()  # inherited; stops the ServerThread
        self.free()  # free liblo resources


# -----------------------------------------------------------------------------
# Channel configuration helpers
# -----------------------------------------------------------------------------

def get_default_channel_config() -> Dict[str, Dict[str, Any]]:
    """Return the textbook Muse‑S channel map used throughout the project."""
    fs_eeg, fs_ppg, fs_acc, fs_metrics, fs_headband_on, fs_hsi, fs_is_good = (
        256, 64, 52, 60, 150, 150, 150
    )

    return {
        "eeg": {"osc_path": "/eeg", "fs": fs_eeg, "ch_slice": None},
        "ppg": {"osc_path": "/ppg", "fs": fs_ppg, "ch_slice": 1},
        "acc": {"osc_path": "/acc", "fs": fs_acc, "ch_slice": slice(0, 3)},
        "metrics": {"osc_path": "/muse_metrics", "fs": fs_metrics, "ch_slice": None},
        "headband_on": {"osc_path": "/headband_on", "fs": fs_headband_on, "ch_slice": None},
        "hsi": {"osc_path": "/hsi", "fs": fs_hsi, "ch_slice": None},
        "is_good": {"osc_path": "/is_good", "fs": fs_is_good, "ch_slice": None},
    }



# -----------------------------------------------------------------------------
# Receiver helpers
# -----------------------------------------------------------------------------

def start_receiver(port: int = 7000, *, lazy: bool = True,
                   logger: logging.Logger = None) -> MuseOSCReceiver:
    """Start a :class:`MuseOSCReceiver` on *port* (skip if already running).

    Parameters
    ----------
    port : int, default 7000
        UDP port to bind.
    lazy : bool, default True
        If *True* (default) and a receiver is already active, return it instead
        of starting a new one.
    """
    if lazy:
        try:
            # If a receiver is already running in the current process, return it
            return MuseOSCReceiver.get_instance()  # type: ignore[attr-defined]
        except AttributeError:
            pass  # no singleton, proceed to create

    if not free_osc_port(port):
        if logger is not None:
            logger.warning("Port %d appears occupied – continuing anyway…", port)  
        else:
            print(f"Warning: Port {port} appears occupied – continuing anyway…") 
    
    rx = MuseOSCReceiver(port=port)
    rx.start()
    if logger is not None:
        logger.info("MuseOSCReceiver started on port %d", port)
    else:
        print(f"MuseOSCReceiver started on port {port}")
        
    return rx

# -----------------------------------------------------------------------------
# Data collection helpers
# -----------------------------------------------------------------------------

def collect_data(
    rx: MuseOSCReceiver,
    channels: Mapping[str, Mapping[str, Any]],
    *,
    duration: float = 10,
    logger: logging.Logger = None,
) -> Dict[str, np.ndarray]:
    """Collect *duration* seconds of data for all *channels*."""
    if logger is not None:
        logger.info("Collecting %.1f s of data…", duration)
    t0 = time.time()
    time.sleep(duration)

    raw: Dict[str, np.ndarray] = {}
    for name, ch in channels.items():
        kwargs = {"ch_slice": ch["ch_slice"]} if ch["ch_slice"] is not None else {}
        raw[name] = rx.get(ch["osc_path"], **kwargs)
    if logger is not None:
        logger.info("Collection done in %.2f s", time.time() - t0)
    return raw


def compute_sample_rates(
    raw: Mapping[str, np.ndarray], channels: Mapping[str, Mapping[str, Any]]
) -> Dict[str, float]:
    """Return actual sample rates for each channel."""
    fps: Dict[str, float] = {}
    for name, ch in channels.items():
        arr = raw.get(name)
        if arr is None or len(arr) < 2:
            fps[name] = np.nan
            continue
        fps[name] = len(arr) / (arr[-1, 0] - arr[0, 0])
    return fps

# -----------------------------------------------------------------------------
# Cleaning helpers
# -----------------------------------------------------------------------------

def regularise_channels(
    raw: Mapping[str, np.ndarray],
    channels: Mapping[str, Mapping[str, Any]],
    *,
    gap_thresh: float = 0.25,
) -> Dict[str, np.ndarray]:
    """Apply :pyfunc:`regularize_timestamps` to any channel with an ``fs``.

    Parameters
    ----------
    gap_thresh : float
        Maximum tolerated gap (seconds) before we *close* the stream and resume
        the timestamp grid. See ``regularize_timestamps`` for details.
    """
    processed: Dict[str, np.ndarray] = {}
    for name, ch in channels.items():
        arr = raw[name]
        if len(arr) > 1 and ch["fs"]:
            processed[name] = regularize_timestamps(arr, fs=ch["fs"], gap_thresh=gap_thresh)
        else:
            processed[name] = arr  # pass‑through
    return processed


# -----------------------------------------------------------------------------
# standalone usage: `python muse_osc_receiver.py` just prints running stats
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    rx = MuseOSCReceiver()
    rx.start()
    print("⇢ MuseOSCReceiver listening on UDP :7000  (Ctrl-C to quit)")
    try:
        while True:
            time.sleep(5)
            eeg = rx.get('/eeg')
            print(f"[{time.strftime('%H:%M:%S')}]   /eeg  buffer = {len(eeg)} samples")
    except KeyboardInterrupt:
        print("\nStopping …")
        rx.stop_gracefully()