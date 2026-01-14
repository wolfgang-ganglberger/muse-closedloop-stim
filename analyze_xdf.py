#!/usr/bin/env python3
"""Load an XDF recording and visualize EEG, virtual frontal, and marker events."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import pyxdf


def _select_stream(streams: List[dict], name: str | None, stype: str | None) -> dict | None:
    if name:
        for stream in streams:
            info = stream.get("info", {})
            if info.get("name", [None])[0] == name:
                return stream
    if stype:
        for stream in streams:
            info = stream.get("info", {})
            if info.get("type", [None])[0] == stype:
                return stream
    return None


def _channel_labels(stream: dict, n_ch: int) -> List[str]:
    info = stream.get("info", {})
    desc = info.get("desc", [])
    if not desc:
        return [f"ch{i+1}" for i in range(n_ch)]
    channels = desc[0].get("channels", [])
    if not channels:
        return [f"ch{i+1}" for i in range(n_ch)]
    channel_entries = channels[0].get("channel", [])
    labels = []
    for idx in range(n_ch):
        if idx < len(channel_entries):
            label = channel_entries[idx].get("label", [f"ch{idx+1}"])[0]
        else:
            label = f"ch{idx+1}"
        labels.append(label)
    return labels


def _bandpass(x: np.ndarray, fs: float, f_lo: float, f_hi: float, order: int = 2) -> np.ndarray:
    nyquist = 0.5 * fs
    b, a = butter(order, [f_lo / nyquist, f_hi / nyquist], btype="bandpass")
    return filtfilt(b, a, x)


def _parse_marker_label(raw_marker: str) -> str:
    parts = str(raw_marker).split(":")
    if len(parts) >= 2 and parts[0] == "stim_sent":
        return parts[1]
    return parts[0]


def _summarize_gaps(t: np.ndarray, fs_est: float) -> Tuple[int, float]:
    if t.size < 2:
        return 0, 0.0
    dt = np.diff(t)
    thresh = 1.5 / fs_est
    gap_mask = dt > thresh
    return int(np.sum(gap_mask)), float(np.max(dt[gap_mask]) if np.any(gap_mask) else 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Muse EEG + markers from an XDF file.")
    parser.add_argument("xdf_path", type=Path, help="Path to .xdf recording")
    parser.add_argument("--eeg-name", default="MuseEEG", help="EEG stream name (default: MuseEEG)")
    parser.add_argument("--marker-name", default="StimMarkers", help="Marker stream name (default: StimMarkers)")
    parser.add_argument("--state-name", default="StimState", help="State stream name (default: StimState)")
    parser.add_argument("--metrics-name", default="MuseMetrics", help="Metrics stream name (default: MuseMetrics)")
    parser.add_argument("--tmin", type=float, default=None, help="Start time (s) relative to EEG stream start")
    parser.add_argument("--tmax", type=float, default=None, help="End time (s) relative to EEG stream start")
    parser.add_argument("--stim-delay", type=float, default=0.2, help="Shift stimulus markers forward by this delay (s)")
    parser.add_argument("--save", type=Path, default=None, help="Save figure to this path instead of showing")
    args = parser.parse_args()

    streams, _ = pyxdf.load_xdf(str(args.xdf_path))

    eeg_stream = _select_stream(streams, args.eeg_name, "EEG")
    marker_stream = _select_stream(streams, args.marker_name, "Markers")
    state_stream = _select_stream(streams, args.state_name, "StimState")
    metrics_stream = _select_stream(streams, args.metrics_name, "Metrics")

    if eeg_stream is None:
        available = [s.get("info", {}).get("name", ["?"])[0] for s in streams]
        raise SystemExit(f"EEG stream not found. Available streams: {available}")

    eeg_ts = np.asarray(eeg_stream["time_stamps"], dtype=float)
    eeg_data = np.asarray(eeg_stream["time_series"], dtype=float)
    if eeg_data.ndim == 1:
        eeg_data = eeg_data[:, None]

    labels = _channel_labels(eeg_stream, eeg_data.shape[1])
    t0 = float(eeg_ts[0])
    t = eeg_ts - t0

    if args.tmin is not None or args.tmax is not None:
        tmin = -np.inf if args.tmin is None else args.tmin
        tmax = np.inf if args.tmax is None else args.tmax
        mask = (t >= tmin) & (t <= tmax)
    else:
        mask = np.ones_like(t, dtype=bool)

    t = t[mask]
    eeg_data = eeg_data[mask]

    dt = np.diff(eeg_ts)
    fs_est = float(1.0 / np.median(dt)) if dt.size > 0 else float("nan")
    gap_count, max_gap = _summarize_gaps(t, fs_est)

    label_to_idx = {lab: idx for idx, lab in enumerate(labels)}
    tp9_idx = label_to_idx.get("TP9", 0)
    tp10_idx = label_to_idx.get("TP10", max(tp9_idx + 1, eeg_data.shape[1] - 1))

    v_frontal = 0.5 * (eeg_data[:, tp9_idx] + eeg_data[:, tp10_idx])
    v_delta = _bandpass(v_frontal, fs_est, 0.5, 4.0, order=2) if np.isfinite(fs_est) else v_frontal

    marker_times: np.ndarray = np.array([])
    marker_labels: List[str] = []
    if marker_stream is not None:
        marker_ts = np.asarray(marker_stream["time_stamps"], dtype=float) - t0 + float(args.stim_delay)
        marker_series = marker_stream["time_series"]
        for ts, item in zip(marker_ts, marker_series):
            raw = item[0] if isinstance(item, (list, tuple, np.ndarray)) else item
            if args.tmin is not None and ts < args.tmin:
                continue
            if args.tmax is not None and ts > args.tmax:
                continue
            marker_times = np.append(marker_times, ts)
            marker_labels.append(_parse_marker_label(raw))

    state_t = np.array([])
    state_data = np.array([])
    state_labels: List[str] = []
    if state_stream is not None:
        state_ts = np.asarray(state_stream["time_stamps"], dtype=float) - t0
        state_series = np.asarray(state_stream["time_series"], dtype=float)
        if state_series.ndim == 1:
            state_series = state_series[:, None]
        if args.tmin is not None or args.tmax is not None:
            smin = -np.inf if args.tmin is None else args.tmin
            smax = np.inf if args.tmax is None else args.tmax
            s_mask = (state_ts >= smin) & (state_ts <= smax)
        else:
            s_mask = np.ones_like(state_ts, dtype=bool)
        state_t = state_ts[s_mask]
        state_data = state_series[s_mask]
        state_labels = _channel_labels(state_stream, state_series.shape[1])

    metrics_t = np.array([])
    metrics_data = np.array([])
    if metrics_stream is not None:
        metrics_ts = np.asarray(metrics_stream["time_stamps"], dtype=float) - t0
        metrics_series = np.asarray(metrics_stream["time_series"], dtype=float)
        if metrics_series.ndim == 1:
            metrics_series = metrics_series[:, None]
        if args.tmin is not None or args.tmax is not None:
            mmin = -np.inf if args.tmin is None else args.tmin
            mmax = np.inf if args.tmax is None else args.tmax
            m_mask = (metrics_ts >= mmin) & (metrics_ts <= mmax)
        else:
            m_mask = np.ones_like(metrics_ts, dtype=bool)
        metrics_t = metrics_ts[m_mask]
        metrics_data = metrics_series[m_mask]

    label_groups: Dict[str, List[float]] = {}
    for ts, lab in zip(marker_times, marker_labels):
        label_groups.setdefault(lab, []).append(float(ts))

    has_state = state_stream is not None and state_data.size
    has_metrics = metrics_stream is not None and metrics_data.size
    n_rows = 3 + (1 if has_metrics else 0) + (1 if has_state else 0)
    height_ratios = [2.5, 2.0, 1.0]
    if has_metrics:
        height_ratios.append(1.0)
    if has_state:
        height_ratios.append(1.0)
    fig, axes = plt.subplots(
        n_rows,
        1,
        sharex=True,
        figsize=(12, 7 + n_rows * 1.0),
        gridspec_kw={"height_ratios": height_ratios},
    )

    if n_rows == 1:
        axes = [axes]
    ax_idx = 0
    ax_eeg = axes[ax_idx]; ax_idx += 1
    ax_vf = axes[ax_idx]; ax_idx += 1
    ax_mark = axes[ax_idx]; ax_idx += 1
    ax_metrics = axes[ax_idx] if has_metrics else None
    if has_metrics:
        ax_idx += 1
    ax_state = axes[ax_idx] if has_state else None
    spacing = np.nanstd(eeg_data) * 3.0 if np.isfinite(np.nanstd(eeg_data)) else 1.0
    offsets = np.arange(eeg_data.shape[1]) * spacing
    for idx, label in enumerate(labels):
        ax_eeg.plot(t, eeg_data[:, idx] + offsets[idx], linewidth=0.7, label=label)
    ax_eeg.set_yticks(offsets)
    ax_eeg.set_yticklabels(labels)
    ax_eeg.set_title("Raw EEG channels (offset for visibility)")
    ax_eeg.grid(True, linewidth=0.3, alpha=0.4)

    ax_vf.plot(t, v_frontal, color="tab:blue", linewidth=0.8, label="virtual frontal (raw)")
    ax_vf.plot(t, v_delta, color="tab:orange", linewidth=1.0, label="virtual frontal (0.5-4 Hz)")
    ax_vf.set_title(f"Virtual frontal (TP9/TP10) | fs≈{fs_est:.2f} Hz, gaps={gap_count}")
    if gap_count > 0:
        ax_vf.set_ylabel(f"max gap {max_gap:.3f}s")
    ax_vf.grid(True, linewidth=0.3, alpha=0.4)
    ax_vf.legend(loc="upper right", fontsize=9)

    if label_groups:
        colors = plt.cm.tab10.colors
        for idx, (lab, times) in enumerate(sorted(label_groups.items())):
            ax_mark.eventplot(
                times,
                lineoffsets=idx + 1,
                linelengths=0.8,
                colors=[colors[idx % len(colors)]],
                linewidths=1.2,
                label=lab,
            )
        ax_mark.set_yticks(range(1, len(label_groups) + 1))
        ax_mark.set_yticklabels([lab for lab, _ in sorted(label_groups.items())])
        ax_mark.legend(loc="upper right", fontsize=8)
    else:
        ax_mark.text(0.5, 0.5, "No markers found", ha="center", va="center", transform=ax_mark.transAxes)
        ax_mark.set_yticks([])

    ax_mark.set_xlabel("Time (s, LSL clock aligned to EEG start)")
    ax_mark.set_title("Stimulus markers")
    ax_mark.grid(True, axis="x", linewidth=0.3, alpha=0.4)

    if ax_metrics is not None:
        sqc_idx = [0, 3]
        stage_idx = [14, 15, 16, 17, 18]
        if metrics_data.shape[1] > max(stage_idx):
            metrics_sqc = metrics_data[:, sqc_idx]
            metrics_stage = metrics_data[:, stage_idx]
            if np.nanmax(metrics_stage) > 1.5:
                metrics_stage = metrics_stage / 100.0
            ax_metrics.plot(metrics_t, metrics_sqc[:, 0], label="SQC CH1", linewidth=1.0, color="tab:gray")
            ax_metrics.plot(metrics_t, metrics_sqc[:, 1], label="SQC CH4", linewidth=1.0, color="tab:blue")
            stage_names = ["Wake", "N1", "N2", "N3", "R"]
            stage_colors = ["tab:orange", "tab:purple", "tab:green", "tab:red", "tab:pink"]
            for idx, (name, color) in enumerate(zip(stage_names, stage_colors)):
                ax_metrics.plot(
                    metrics_t,
                    metrics_stage[:, idx],
                    label=f"P({name})",
                    linewidth=1.0,
                    color=color,
                    alpha=0.9,
                )
            ax_metrics.set_ylim(0.0, 1.05)
            ax_metrics.set_ylabel("Probability")
            ax_metrics.set_title("Signal quality (CH1/CH4) + sleep stage probabilities")
            ax_metrics.grid(True, axis="x", linewidth=0.3, alpha=0.4)
            ax_metrics.legend(loc="upper right", fontsize=8, ncol=4)
        else:
            ax_metrics.text(0.5, 0.5, "Metrics stream missing expected channels", ha="center", va="center", transform=ax_metrics.transAxes)
            ax_metrics.set_yticks([])

    if ax_state is not None:
        for idx, label in enumerate(state_labels):
            ax_state.plot(
                state_t,
                state_data[:, idx],
                drawstyle="steps-post",
                linewidth=1.0,
                label=label,
            )
        ax_state.set_ylim(-0.1, 1.1)
        ax_state.set_title("Stim state (global gate + window)")
        ax_state.set_yticks([0, 1])
        ax_state.grid(True, axis="x", linewidth=0.3, alpha=0.4)
        ax_state.legend(loc="upper right", fontsize=8, ncol=2)

    fig.suptitle(args.xdf_path.name, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    overlay_fig = plt.figure(figsize=(12, 4))
    ax_overlay = overlay_fig.add_subplot(1, 1, 1)
    ax_overlay.plot(
        t,
        eeg_data[:, tp9_idx],
        color="gray",
        alpha=0.5,
        linewidth=0.8,
        label=labels[tp9_idx],
    )
    if tp10_idx != tp9_idx and tp10_idx < eeg_data.shape[1]:
        ax_overlay.plot(
            t,
            eeg_data[:, tp10_idx],
            color="gray",
            alpha=0.5,
            linewidth=0.8,
            label=labels[tp10_idx],
        )
    ax_overlay.plot(t, v_frontal, color="black", linewidth=1.0, label="virtual frontal (raw)")
    ax_overlay.plot(t, v_delta, color="tab:blue", linewidth=1.2, label="virtual frontal (0.5-4 Hz)")
    if marker_times.size:
        for ts in marker_times:
            ax_overlay.axvline(ts, color="red", alpha=0.9, linewidth=1.0)
    ax_overlay.set_xlabel("Time (s, LSL clock aligned to EEG start)")
    ax_overlay.set_ylabel("Amplitude (uV)")
    ax_overlay.set_title("Temporal channels + virtual frontal + markers")
    ax_overlay.grid(True, linewidth=0.3, alpha=0.4)
    ax_overlay.legend(loc="upper right", fontsize=9, ncol=2)
    overlay_fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150)
        overlay_path = args.save.with_name(f"{args.save.stem}_overlay{args.save.suffix}")
        overlay_fig.savefig(overlay_path, dpi=150)
    else:
        plt.show()


if __name__ == "__main__":
    main()
