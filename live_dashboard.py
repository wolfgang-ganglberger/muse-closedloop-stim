"""live_dashboard.py
===================
Reusable utilities for collecting, cleaning, and visualising Muse‑S OSC data.

Typical usage from a notebook::

    import live_dashboard as ld

    rx = ld.start_receiver(port=7000)
    channels = ld.get_default_channel_config()

    raw = ld.collect_data(rx, channels, duration=10)
    processed = ld.regularise_channels(raw, channels)
    ld.plot_static_overview(processed, title="10‑second snapshot")

    # Launch live dashboard – interrupt (Ctrl‑C) to stop
    ld.launch_live_dashboard(rx, channels)

Author: Wolfgang Ganglberger, 2025.
"""

from __future__ import annotations

import time
import logging
from typing import Dict, Any, Mapping

import numpy as np
import matplotlib.pyplot as plt

try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtWidgets
except ImportError:  # runtime‑safe; dashboard only works if PyQtGraph present
    pg = None  # type: ignore
    QtWidgets = None  # type: ignore

import sys
sys.path.append('./closedloop-utils')
from osc_receiver import MuseOSCReceiver, free_osc_port, start_receiver
from osc_receiver import get_default_channel_config, collect_data, compute_sample_rates, regularise_channels

__all__ = [
    "MuseOSCReceiver",
    "free_osc_port",
    "start_receiver",
    "get_default_channel_config",
    "collect_data",
    "compute_sample_rates",
    "regularise_channels",
]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------


def init_colors(background: str = "white") -> Dict[str, str]:
    """Initialize a dictionary of colors for Muse channels."""
    
    colors = {
        "AF7": "black",
        "TP9": "darkblue",
        "TP10": "blue",
        "AF8": "gray",
        
        "PPG IR": "crimson",
        
        "ACC X": "lightgreen",
        "ACC Y": "limegreen",
        "ACC Z": "darkgreen",
    }
    
    if background == "black":
        # Dark mode colors
        colors = {
            "AF7": "gold",
            "TP9": "gold",
            "TP10": "gold",
            "AF8": "gold",
            
            "PPG IR": "crimson",
            
            "ACC X": "lightgreen",
            "ACC Y": "limegreen",
            "ACC Z": "darkgreen",
        }
        
    return colors


def plot_static_overview(
    data: Mapping[str, np.ndarray],
    *,
    title: str | None = None,
    eeg_offset: float = 100.0,
):
    """Matplotlib snapshot of EEG, PPG (IR Ch1) and ACC."""
    eeg = data["eeg"]
    ppg = data["ppg"]
    acc = data["acc"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # --- EEG
    eeg_names = ["AF7", "TP9", "TP10", "AF8"]
    colors = init_colors()
    ch_order = [0, 3, 1, 2]  # plot in the order: AF7, AF8, TP9, TP10. I.e. Double frontal, then tempo-parietal
    for i, ch_idx in enumerate(ch_order):
        ch_name = eeg_names[ch_idx]
        ch_data = eeg[:, ch_idx + 1]  # skip timestamp column
        ch_color = colors[ch_name]

        axes[0].plot(
            eeg[:, 0] - eeg[0, 0],
            ch_data - eeg_offset * i,
            label=ch_name,
            color=ch_color
        )
        
    axes[0].set_ylabel("EEG (µV, offset)")
    axes[0].legend(loc="upper right")
    axes[0].set_title("EEG")

    # --- PPG IR
    if ppg.shape[1] > 1:
        axes[1].plot(ppg[:, 0] - ppg[0, 0], ppg[:, 1], color=colors["PPG IR"])
    axes[1].set_ylabel("PPG IR (a.u.)")
    axes[1].set_title("PPG IR (Ch1)")

    # --- ACC
    acc_names = ["X", "Y", "Z"]
    for ch in range(1, min(4, acc.shape[1])):
        axes[2].plot(acc[:, 0] - acc[0, 0], acc[:, ch], label=f"ACC {acc_names[ch - 1]}", color=colors[f"ACC {acc_names[ch - 1]}"])
    axes[2].set_ylabel("Accelerometer (g)")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(loc="upper right")
    axes[2].set_title("Accelerometer")

    if title:
        plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------
# Live dashboard (PyQtGraph)
# -----------------------------------------------------------------------------

def _require_pyqtgraph():
    if pg is None or QtWidgets is None:
        raise ImportError("pyqtgraph/Qt not available – install to use live dashboard")


def launch_live_dashboard(
    rx: MuseOSCReceiver,
    channels: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    window_sec: float = 10.0,
    eeg_offset: float = 100.0,
    ppg_channel: int = 1,
    refresh_dt: float = 0.05,
):
    """Launch a stacked‑plot dashboard (EEG · PPG · ACC) that updates live.

    ``Ctrl‑C`` in terminal or close the Qt window to exit.
    """
    _require_pyqtgraph()
    channels = channels or get_default_channel_config()

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = pg.GraphicsLayoutWidget(title="Live Muse‑S Data")
    win.show()

    # stacked sub‑plots
    eeg_plot = win.addPlot(row=0, col=0, title="EEG 4‑ch")
    ppg_plot = win.addPlot(row=1, col=0, title="PPG (IR)")
    acc_plot = win.addPlot(row=2, col=0, title="Accelerometer")
    sqc_plot = win.addPlot(row=3, col=0, title="SQC (Muse Metrics)")
    stage_plot = win.addPlot(row=4, col=0, title="Sleep Stage % (Muse Metrics)")

    win.ci.layout.setRowStretchFactor(0, 3)
    win.ci.layout.setRowStretchFactor(1, 1)
    win.ci.layout.setRowStretchFactor(2, 1)
    win.ci.layout.setRowStretchFactor(3, 1)
    win.ci.layout.setRowStretchFactor(4, 1)

    raw_eeg_names = ["TP9", "AF7", "AF8", "TP10"]
    ch_order = [0, 3, 1, 2]  # plot in the order: TP9, TP10, AF7, AF8
    eeg_names = [raw_eeg_names[ch] for ch in ch_order]
    
    acc_names = ["X", "Y", "Z"]
    colors = init_colors(background="black")
    sqc_colors = ["#9e9e9e", "#7fb3d5", "#76d7c4", "#f7dc6f"]
    stage_names = ["Wake", "N1", "N2", "N3", "R"]
    stage_colors = ["#f4d03f", "#a569bd", "#5dade2", "#58d68d", "#ec7063"]
    
    eeg_curves = [eeg_plot.plot(pen=pg.mkPen(colors[n]), name=n) for n in eeg_names]
    eeg_plot.addLegend(offset=(10, 10))
    eeg_plot.showGrid(y=True)

    ppg_curve = ppg_plot.plot(pen=pg.mkPen(colors["PPG IR"]))
    acc_curves = [acc_plot.plot(pen=pg.mkPen(colors[f"ACC {acc_names[ch - 1]}"])) for ch in range(1, 4)]
    sqc_curves = [
        sqc_plot.plot(pen=pg.mkPen(sqc_colors[i]), name=f"SQC CH{i + 1}")
        for i in range(4)
    ]
    sqc_plot.addLegend(offset=(10, 10))
    sqc_plot.setYRange(0, 100)

    stage_curves = [
        stage_plot.plot(pen=pg.mkPen(stage_colors[i]), name=stage_names[i])
        for i in range(5)
    ]
    stage_plot.addLegend(offset=(10, 10))
    stage_plot.setYRange(0, 100)

    try:
        while True:
            # EEG
            eeg = rx.get(channels["eeg"]["osc_path"])[:, :5]
            if eeg.size:
                t, sig = eeg[:, 0], eeg[:, 1:].T
                m = t > (t[-1] - window_sec)
                x = t[m] - t[m][-1]
                for idx, curve in enumerate(eeg_curves):
                    ch = ch_order[idx]
                    curve.setData(x, sig[ch][m] - eeg_offset * idx)
                    
            # PPG
            ppg = rx.get(channels["ppg"]["osc_path"])
            if ppg.size:
                t, sig = ppg[:, 0], ppg[:, 1:].T
                m = t > (t[-1] - window_sec)
                x = t[m] - t[m][-1]
                ppg_curve.setData(x, sig[ppg_channel][m])

            # ACC
            acc = rx.get(channels["acc"]["osc_path"])
            if acc.size:
                t, sig = acc[:, 0], acc[:, 1:].T
                m = t > (t[-1] - window_sec)
                x = t[m] - t[m][-1]
                for ch, c in enumerate(acc_curves):
                    c.setData(x, sig[ch][m])

            # Muse metrics (SQC + sleep stages)
            metrics = rx.get(channels["metrics"]["osc_path"])
            if metrics.size:
                t = metrics[:, 0]
                sig = metrics[:, 1:].T
                m = t > (t[-1] - window_sec)
                x = t[m] - t[m][-1]

                # SQC channels 0-3 -> columns 1-4 in the metrics packet.
                for idx, c in enumerate(sqc_curves):
                    c.setData(x, sig[idx][m])

                # Sleep stage percentages 14-18 -> columns 15-19 in the packet.
                for idx, c in enumerate(stage_curves):
                    c.setData(x, sig[14 + idx][m])

            app.processEvents()
            time.sleep(refresh_dt)
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user")
        rx.stop_gracefully()
        win.close()  # type: ignore[attr-defined]

# -----------------------------------------------------------------------------
# Script mode – quick demo: Collect and plot a 5‑second snapshot, then launch live dashboard
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    rx = start_receiver(logger=logger)
    chans = get_default_channel_config()

    raw = collect_data(rx, chans, duration=5, logger=logger)
    fps = compute_sample_rates(raw, chans)
    for n, f in fps.items():
        logger.info("%s: %.1f Hz (expected %s)" % (n.upper(), f, chans[n]["fs"]))

    processed = regularise_channels(raw, chans)
    plot_static_overview(processed, title="5‑second snapshot")

    print("Launching live dashboard – Ctrl‑C to quit…")
    launch_live_dashboard(rx, chans)
