# --- LSL support ---
import threading
from pylsl import StreamInfo, StreamOutlet, local_clock
import pylsl
print("Using pylsl version:", pylsl.__version__)
import time
from typing import Dict, Any, Mapping
from osc_receiver import MuseOSCReceiver
from osc_receiver import free_osc_port, start_receiver
import logging
import numpy as np

# LSL streaming helpers
# -----------------------------------------------------------------------------

class _EEGLSLStreamer(threading.Thread):
    """Background thread that streams Muse‑S EEG samples into an LSL outlet.

    Parameters
    ----------
    rx : MuseOSCReceiver
        The active OSC receiver instance.
    name : str
        Name of the LSL stream shown in LabRecorder.
    channel_names : tuple[str, ...]
        Labels for the EEG channels in original headband order.
    fs : int
        Nominal sampling rate of the Muse EEG (default 256 Hz).
    chunk_size : int
        How many samples to push in one LSL chunk (≤1024 recommended).
    ch_slice : slice | tuple[int, ...] | None
        Slice or tuple of column indices to select channels from the raw data.
        If None, selects all columns corresponding to channel_names.
    
    Note: pylsl.StreamOutlet.push_chunk expects the samples list as first positional argument and the per‑sample timestamp list as the second positional argument; keyword args are not supported in pylsl ≤1.16.
        Data are cast to float32 before streaming.
        
    """
    def __init__(
        self,
        rx: 'MuseOSCReceiver',
        *,
        name: str = "MuseEEG",
        channel_names: tuple[str, ...] = ("TP9", "AF7", "AF8", "TP10", "AUX_1", "AUX_2"),
        fs: int = 256,
        chunk_size: int = 32,
        ch_slice: slice | tuple[int, ...] | None = None
    ):

        super().__init__(daemon=True)
        self.rx = rx
        self.fs = fs
        self.chunk_size = chunk_size
        self.channel_names = channel_names
        
        self.ch_slice = ch_slice
        # fall back to all columns if no slice provided
        if self.ch_slice is None:
            self.ch_slice = slice(0, len(channel_names))
        if isinstance(self.ch_slice, slice):
            expected_cols = (self.ch_slice.stop or len(self.channel_names)) - (self.ch_slice.start or 0)
        else:
            expected_cols = len(self.ch_slice)
        assert expected_cols == len(self.channel_names), (
            "channel_names length must match number of columns selected by ch_slice"
        )
        
        self._stop = threading.Event()
        self._last_idx = 0  # points to first *unstreamed* row in rx buffer
        
        # ---- build LSL outlet ----
        info = StreamInfo(name, "EEG", len(self.channel_names), fs, "float32", "muse_eeg_stream")
        chns = info.desc().append_child("channels")
        for lab in self.channel_names:
            ch = chns.append_child("channel")
            ch.append_child_value("label", lab)
        self.outlet = StreamOutlet(info, chunk_size=chunk_size, max_buffered=fs * 60)

    # -----------------------------------------------------------------
    def run(self):
        while not self._stop.is_set():
            buf = self.rx.get("/eeg", ch_slice=self.ch_slice)

            if buf.size == 0:
                time.sleep(0.01)
                continue

            # Track last streamed timestamp
            if not hasattr(self, '_last_ts'):
                self._last_ts = buf[-1, 0]

            # Find all new rows
            new_rows = buf[buf[:, 0] > self._last_ts]
            if new_rows.shape[0] > 0:
                data_cols = new_rows[:, 1:1+len(self.channel_names)].astype(np.float32)
                ts_list = new_rows[:, 0].tolist()
                self.outlet.push_chunk(data_cols.tolist(), ts_list)
                self._last_ts = ts_list[-1]
                print(f"LSL pushed chunk: {len(data_cols)} samples, last timestamp {ts_list[-1]:.3f}")
            else:
                print(f"No new data to push. Last timestamp: {self._last_ts:.3f}")
                time.sleep(0.01)


    def stop(self):
        self._stop.set()


class _MetricsLSLStreamer(threading.Thread):
    """Background thread that streams Muse‑S metrics samples into an LSL outlet."""
    def __init__(
        self,
        rx: 'MuseOSCReceiver',
        *,
        name: str = "MuseMetrics",
        channel_names: tuple[str, ...] = tuple(f"metric_{i:02d}" for i in range(27)),
        fs: int = 60,
        chunk_size: int = 10,
        ch_slice: slice | tuple[int, ...] | None = None,
        osc_path: str = "/muse_metrics",
        stream_type: str = "Metrics",
    ):
        super().__init__(daemon=True)
        self.rx = rx
        self.fs = fs
        self.chunk_size = chunk_size
        self.channel_names = channel_names
        self.osc_path = osc_path

        self.ch_slice = ch_slice
        if self.ch_slice is None:
            self.ch_slice = slice(0, len(channel_names))
        if isinstance(self.ch_slice, slice):
            expected_cols = (self.ch_slice.stop or len(self.channel_names)) - (self.ch_slice.start or 0)
        else:
            expected_cols = len(self.ch_slice)
        assert expected_cols == len(self.channel_names), (
            "channel_names length must match number of columns selected by ch_slice"
        )

        self._stop = threading.Event()

        info = StreamInfo(name, stream_type, len(self.channel_names), fs, "float32", "muse_metrics_stream")
        chns = info.desc().append_child("channels")
        for lab in self.channel_names:
            ch = chns.append_child("channel")
            ch.append_child_value("label", lab)
        self.outlet = StreamOutlet(info, chunk_size=chunk_size, max_buffered=fs * 60)

    def run(self):
        while not self._stop.is_set():
            buf = self.rx.get(self.osc_path, ch_slice=self.ch_slice)
            if buf.size == 0:
                time.sleep(0.05)
                continue

            if not hasattr(self, "_last_ts"):
                self._last_ts = buf[-1, 0]

            new_rows = buf[buf[:, 0] > self._last_ts]
            if new_rows.shape[0] > 0:
                data_cols = new_rows[:, 1:1+len(self.channel_names)].astype(np.float32)
                ts_list = new_rows[:, 0].tolist()
                self.outlet.push_chunk(data_cols.tolist(), ts_list)
                self._last_ts = ts_list[-1]
            else:
                time.sleep(0.05)

    def stop(self):
        self._stop.set()


def start_eeg_lsl_stream(rx: 'MuseOSCReceiver',
                         channel_names: tuple[str, ...] | None = None,
                         ch_slice: slice | tuple[int, ...] | None = None,
                         logger: logging.Logger = None,
                         **kwargs) -> _EEGLSLStreamer:
    """Convenience wrapper to begin streaming Muse EEG into LSL.

    Example
    -------
    >>> rx = start_receiver()
    >>> lsl_thread = start_eeg_lsl_stream(rx)
    >>> # … run your closed‑loop / LabRecorder …
    >>> lsl_thread.stop()
    
    To stream only all six channels:

    >>> lsl = start_eeg_lsl_stream(rx, channel_names=("TP9","AF7","AF8","TP10","AUX_1","AUX_2"),
    ...                            ch_slice=slice(0,6))
    Default: 4 EEG channels only.
    """
    
    if channel_names is None:
        channel_names = ("TP9", "AF7", "AF8", "TP10")
        print('EEG SLS Streamer, de fault EEG channel selected: ', channel_names)

    streamer = _EEGLSLStreamer(rx, channel_names=channel_names, ch_slice=ch_slice, **kwargs)
    streamer.start()
    if logger is not None:
        logger.info("Started LSL streamer '%s' (%d Hz)", kwargs.get("name", "MuseEEG"), kwargs.get("fs", 256))
    return streamer


def start_metrics_lsl_stream(
    rx: "MuseOSCReceiver",
    channel_names: tuple[str, ...] | None = None,
    ch_slice: slice | tuple[int, ...] | None = None,
    logger: logging.Logger = None,
    **kwargs,
) -> _MetricsLSLStreamer:
    """Convenience wrapper to begin streaming Muse metrics into LSL."""
    if channel_names is None:
        channel_names = tuple(f"metric_{i:02d}" for i in range(30))
    streamer = _MetricsLSLStreamer(rx, channel_names=channel_names, ch_slice=ch_slice, **kwargs)
    streamer.start()
    if logger is not None:
        logger.info("Started LSL streamer '%s' (%d Hz)", kwargs.get("name", "MuseMetrics"), kwargs.get("fs", 60))
    return streamer

# -----------------------------------------------------------------------------

def main():
        
    """Quick demo: Collect and stream EEG data into LSL."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    # Start the OSC receiver
    rx = start_receiver()
    
    # Start the LSL streamer
    print("Starting LSL streamer for EEG data…")
    lsl_streamer = start_eeg_lsl_stream(rx) # ← EEG now visible in LabRecorder
    print("LSL streamer started. Press Ctrl+C to stop.")
    # TMP: sleep for 5 seconds:
    time.sleep(5)
    print('hello world, lets also start streaming from OSC!')
    try:
        # Keep the main thread alive while streaming
        while True:
            time.sleep(1)
            # print an EEG value:
            rx_data = rx.get("/eeg", ch_slice=slice(0, 4))
            if rx_data.size > 0:
                print(f"EEG sample: {rx_data[-1, 1:5]}")
            else:
                print("No EEG data available yet.")
    except KeyboardInterrupt:
        print("Stopping LSL streamer…")
        lsl_streamer.stop()
        print("Done.")

if __name__ == "__main__":
    main()
