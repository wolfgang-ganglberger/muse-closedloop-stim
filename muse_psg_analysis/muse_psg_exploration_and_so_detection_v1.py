"""Quick exploration of simultaneous Muse and PSG recordings."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import random

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SO_DETECTION_DIR = REPO_ROOT / "so_detection_dev"
if str(SO_DETECTION_DIR) not in sys.path:
    sys.path.insert(0, str(SO_DETECTION_DIR))

from run_sw_detection_pipeline import evaluate_targeting, prepare_virtual_channels
from virtual_frontal import lowpass_delta_frontal
from so_detection_dev.sw_detection_utils import detect_so_events


DEFAULT_DATA_DIR = Path(
    "/Users/wolfgang/cdac Dropbox/a_People_BIDMC/WolfgangGanglberger/muse_data_scn_montreal/Simultaneous Muse-PSG/"
)
TARGET_FS = 256.0
RESULTS_DIR = REPO_ROOT / "so_results"


@dataclass
class PreparedSignals:
    sid: str
    base_dir: Path
    fs: float
    overlap_sec: float
    muse_df_raw: pd.DataFrame
    muse_df_prepped: pd.DataFrame
    psg_picks: list[str]
    psg_leads_uv: np.ndarray
    v_psg: np.ndarray
    v_psg_filt: np.ndarray
    v_muse: np.ndarray
    v_muse_filt: np.ndarray


def describe_raw(raw: mne.io.BaseRaw, label: str) -> None:
    print(f"\n{label} info:")
    filenames = getattr(raw, "filenames", None) or []
    if filenames:
        print(f"  File: {filenames[0]}")
    print(f"  Channels ({len(raw.ch_names)}): {', '.join(raw.ch_names)}")
    print(f"  Sampling rate: {raw.info['sfreq']:.2f} Hz")
    duration_min = raw.n_times / raw.info["sfreq"] / 60.0
    print(f"  Duration: {duration_min:.2f} min")
    meas_date = raw.info.get("meas_date")
    if meas_date:
        print(f"  Measurement start: {meas_date}")


def resample_to_target(raw: mne.io.BaseRaw, label: str) -> mne.io.BaseRaw:
    fs_in = float(raw.info["sfreq"])
    if not np.isclose(fs_in, TARGET_FS):
        print(f"Resampling {label} from {fs_in:.2f} Hz to {TARGET_FS:.2f} Hz")
        raw = raw.copy().resample(TARGET_FS, npad="auto")
    else:
        print(f"{label} already at {TARGET_FS:.2f} Hz; cloning data for consistency")
        raw = raw.copy()
    return raw


def load_muse_dataframe(raw: mne.io.BaseRaw) -> pd.DataFrame:
    eeg_channels = [name for name in raw.ch_names if name.lower().startswith("eegch")]
    eeg_channels.sort(key=lambda name: int("".join(filter(str.isdigit, name)) or 0))

    if len(eeg_channels) < 4:
        raise RuntimeError(
            f"Expected at least four Muse EEG channels named eegch*, found: {', '.join(raw.ch_names)}"
        )

    data = {}
    for idx, real_name in enumerate(eeg_channels[:4]):
        alias = f"ch{idx + 1}"
        values = raw.get_data(picks=[real_name])[0] * 1e6
        data[alias] = values

    times = raw.times
    df = pd.DataFrame(data, index=pd.Index(times, name="t_sec"))
    df = df[["ch1", "ch2", "ch3", "ch4"]]
    df["noise_level_channel_1"] = 0.2
    df["noise_level_channel_4"] = 0.2
    df["pred_stage"] = "N2"
    return df


def pick_frontal_channels(raw: mne.io.BaseRaw, keywords: Iterable[str]) -> list[str]:
    selected = []
    upper_keywords = [k.upper() for k in keywords]
    for name in raw.ch_names:
        comp = name.upper()
        if any(key in comp for key in upper_keywords):
            selected.append(name)
    return selected


def _to_datetime(value):
    if value is None:
        return None
    return pd.Timestamp(value).to_pydatetime()


def align_recordings(
    raw_muse: mne.io.BaseRaw,
    raw_psg: mne.io.BaseRaw,
) -> Tuple[mne.io.BaseRaw, mne.io.BaseRaw, float]:
    muse_start = _to_datetime(raw_muse.info.get("meas_date"))
    psg_start = _to_datetime(raw_psg.info.get("meas_date"))

    if muse_start is None or psg_start is None:
        print("Could not determine measurement start times; skipping alignment.")
        return raw_muse, raw_psg, 0.0

    start_delta = (psg_start - muse_start).total_seconds()
    muse_offset = max(0.0, start_delta)
    psg_offset = max(0.0, -start_delta)

    muse_total = raw_muse.times[-1]
    psg_total = raw_psg.times[-1]
    muse_available = muse_total - muse_offset
    psg_available = psg_total - psg_offset
    overlap_sec = max(0.0, min(muse_available, psg_available))

    if overlap_sec <= 0.0:
        raise RuntimeError("No temporal overlap between Muse and PSG recordings")

    print(
        f"Aligning recordings: muse_offset={muse_offset:.2f}s, psg_offset={psg_offset:.2f}s, "
        f"overlap={overlap_sec / 60.0:.2f} min"
    )

    muse_aligned = raw_muse.copy().crop(tmin=muse_offset, tmax=muse_offset + overlap_sec, include_tmax=False)
    psg_aligned = raw_psg.copy().crop(tmin=psg_offset, tmax=psg_offset + overlap_sec, include_tmax=False)
    return muse_aligned, psg_aligned, overlap_sec


def prepare_virtual_signals(
    sid: str,
    base_dir: Path,
    *,
    verbose: bool = True,
) -> PreparedSignals:
    muse_path = base_dir / f"{sid}" / "muse" / f"{sid}_muse.edf"
    psg_path = base_dir / f"{sid}" / "PSG" / f"{sid}_psg.edf"

    if verbose:
        print(f"Subject: {sid}")
        print(f"Data dir: {base_dir}")
        print("Loading Muse EDF...")
    raw_muse = mne.io.read_raw_edf(muse_path, preload=True, verbose="ERROR")
    if verbose:
        describe_raw(raw_muse, "Muse")
    raw_muse = resample_to_target(raw_muse, "Muse")
    raw_muse.filter(l_freq=0.1, h_freq=20.0, picks="eeg", verbose="ERROR")

    if verbose:
        print("Loading PSG EDF...")
    raw_psg = mne.io.read_raw_edf(psg_path, preload=True, verbose="ERROR")
    if verbose:
        describe_raw(raw_psg, "PSG")
    raw_psg = resample_to_target(raw_psg, "PSG")
    raw_psg.filter(l_freq=0.3, h_freq=20.0, picks="eeg", verbose="ERROR")

    raw_muse, raw_psg, overlap_sec = align_recordings(raw_muse, raw_psg)

    muse_df = load_muse_dataframe(raw_muse)
    fs = float(raw_muse.info["sfreq"])
    if verbose:
        print("\nMuse DataFrame snapshot:")
        print(muse_df.head())
        print(muse_df.describe().loc[["mean", "std", "min", "max"]])

    df_prepped = prepare_virtual_channels(muse_df.reset_index(drop=False), fs=fs)
    if verbose:
        print("\nPrepared Muse columns:")
        cols = ["v_frontal", "v1", "v2", "v_frontal_filt", "mask_good"]
        print(df_prepped.loc[:, cols].head())

    v_muse = df_prepped["v_frontal"].to_numpy(dtype=float)
    v_muse_filt = df_prepped["v_frontal_filt"].to_numpy(dtype=float)

    frontal_keywords = ("F4:M1", "F3:M2")
    psg_picks = pick_frontal_channels(raw_psg, frontal_keywords)
    if not psg_picks:
        raise RuntimeError("No PSG frontal channels matched the default keywords")
    if verbose:
        print(f"\nUsing PSG channels for comparison: {', '.join(psg_picks)}")

    psg_raw_frontals = raw_psg.copy().pick(psg_picks)
    psg_data = psg_raw_frontals.get_data() * 1e6
    v_psg = psg_data.mean(axis=0)
    v_psg_filt = lowpass_delta_frontal(v_psg, fs=fs)
    print("For now inverting PSG virtual frontal to match Muse polarity. Not clear why they are opposite here. Deconvolution?")
    v_psg_filt = -1 * v_psg_filt
    
    return PreparedSignals(
        sid=sid,
        base_dir=base_dir,
        fs=fs,
        overlap_sec=overlap_sec,
        muse_df_raw=muse_df,
        muse_df_prepped=df_prepped,
        psg_picks=psg_picks,
        psg_leads_uv=psg_data,
        v_psg=v_psg,
        v_psg_filt=v_psg_filt,
        v_muse=v_muse,
        v_muse_filt=v_muse_filt,
    )


def compare_muse_to_psg(
    prepared: PreparedSignals,
    plot_window_sec: float = 30.0,
    *,
    verbose: bool = False,
) -> None:
    fs = prepared.fs
    overlap_sec = prepared.overlap_sec
    psg_picks = prepared.psg_picks

    v_frontal = prepared.v_muse.copy()
    v_frontal_filt = prepared.v_muse_filt.copy()
    v_psg = prepared.v_psg.copy()
    v_psg_filt = prepared.v_psg_filt.copy()
    psg_data = prepared.psg_leads_uv.copy()

    if verbose:
        print("\nMuse DataFrame snapshot:")
        print(prepared.muse_df_raw.head())
        print(prepared.muse_df_raw.describe().loc[["mean", "std", "min", "max"]])
        print("\nPrepared Muse columns:")
        cols = ["v_frontal", "v1", "v2", "v_frontal_filt", "mask_good"]
        print(prepared.muse_df_prepped.loc[:, cols].head())
        print(f"\nUsing PSG channels for comparison: {', '.join(psg_picks)}")

    min_len = min(
        len(v_frontal),
        len(v_frontal_filt),
        len(v_psg),
        len(v_psg_filt),
        psg_data.shape[1],
    )
    if len(v_frontal) != min_len:
        v_frontal = v_frontal[:min_len]
    if len(v_frontal_filt) != min_len:
        v_frontal_filt = v_frontal_filt[:min_len]
    if len(v_psg) != min_len:
        v_psg = v_psg[:min_len]
    if len(v_psg_filt) != min_len:
        v_psg_filt = v_psg_filt[:min_len]
    if psg_data.shape[1] != min_len:
        psg_data = psg_data[:, :min_len]

    n_samples = int(plot_window_sec * fs)
    n_samples = min(n_samples, min_len)
    if n_samples == 0:
        raise RuntimeError("Not enough samples to plot comparison")

    snippet_start = max(0, (min_len - n_samples) // 2)
    snippet_stop = snippet_start + n_samples

    time_axis = np.arange(n_samples) / fs
    total_overlap_min = overlap_sec / 60.0 if overlap_sec else min_len / fs / 60.0

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(10, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1, 1]},
    )

    ax_psg_leads, ax_psg_virtual, ax_muse, ax_compare = axes

    for idx, ch_name in enumerate(psg_picks):
        ax_psg_leads.plot(
            time_axis,
            psg_data[idx, snippet_start:snippet_stop],
            label=f"PSG {ch_name}",
            linewidth=0.9,
        )
    ax_psg_leads.set_ylabel("PSG (µV)")
    ax_psg_leads.set_title(
        "PSG frontal leads"
        f" ({plot_window_sec:g}s excerpt of {total_overlap_min:.1f} min overlap)"
    )
    ax_psg_leads.grid(True, linewidth=0.3, alpha=0.4)
    ax_psg_leads.legend(loc="upper right", fontsize=9)

    ax_psg_virtual.plot(
        time_axis,
        v_psg[snippet_start:snippet_stop],
        label="PSG virtual frontal (raw)",
        linewidth=1.0,
    )
    ax_psg_virtual.plot(
        time_axis,
        v_psg_filt[snippet_start:snippet_stop],
        label="PSG virtual frontal (delta)",
        linewidth=0.9,
    )
    ax_psg_virtual.set_ylabel("PSG virtual (µV)")
    ax_psg_virtual.set_title("PSG virtual frontal average")
    ax_psg_virtual.grid(True, linewidth=0.3, alpha=0.4)
    ax_psg_virtual.legend(loc="upper right", fontsize=9)

    ax_muse.plot(
        time_axis,
        v_frontal[snippet_start:snippet_stop],
        label="Muse v_frontal (raw)",
        linewidth=1.0,
    )
    ax_muse.plot(
        time_axis,
        v_frontal_filt[snippet_start:snippet_stop],
        label="Muse v_frontal_filt",
        linewidth=0.9,
    )
    ax_muse.set_xlabel("Time (s)")
    ax_muse.set_ylabel("Muse (µV)")
    ax_muse.set_title("Muse virtual frontal")
    ax_muse.grid(True, linewidth=0.3, alpha=0.4)
    ax_muse.legend(loc="upper right", fontsize=9)

    ax_compare.plot(
        time_axis,
        v_psg_filt[snippet_start:snippet_stop],
        color="black",
        linewidth=1.0,
        label="PSG virtual (delta) × -1",
    )
    ax_compare.plot(
        time_axis,
        v_frontal_filt[snippet_start:snippet_stop],
        color="red",
        linewidth=1.0,
        label="Muse virtual (delta)",
    )
    ax_compare.set_xlabel("Time (s)")
    ax_compare.set_ylabel("µV")
    ax_compare.set_title("Muse vs. PSG virtual frontal (delta)")
    ax_compare.grid(True, linewidth=0.3, alpha=0.4)
    ax_compare.legend(loc="upper right", fontsize=9)
    
    # add minor ticks for 1-second intervals
    ax_compare.set_xticks(np.arange(0, plot_window_sec + 1, 1.0), minor=True)
    ax_compare.grid(which='minor', linestyle='--', linewidth=0.2, alpha=0.3)

    fig.tight_layout()
    plt.show()


def _save_events(df: pd.DataFrame, file_name: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / file_name
    df.to_csv(out_path, index=False)
    print(f"Saved slow-wave detections to {out_path}")
    return out_path


def detect_slow_waves_offline(
    prepared: PreparedSignals,
    preset: str = "ngo2013",
) -> tuple[pd.DataFrame, dict]:
    """Detect slow oscillations offline using the virtual PSG channel."""

    fs = prepared.fs
    n = len(prepared.v_psg)
    time_axis = np.arange(n) / fs

    df_psg = pd.DataFrame(
        {
            "t_sec": time_axis,
            "v_psg": prepared.v_psg,
            "v_psg_filt": prepared.v_psg_filt,
            "mask_good": np.ones(n, dtype=bool),
            "sleep_stage": "N2",
        }
    )

    events, summary = detect_so_events(
        df_psg,
        fs=fs,
        signal_col="v_psg_filt",
        sigma_carrier_col="v_psg",
        preset=preset,
        stages=None,
    )

    print('##########################################')
    print("Offline detection summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    _save_events(events, f"so_detection_offline_{prepared.sid}.csv")
    return events, summary


def detect_slow_waves_online(
    prepared: PreparedSignals,
    preset: str = "ngo2013",
) -> tuple[pd.DataFrame, dict]:
    """Mock online slow-wave detector using the Muse virtual channel.

    TODO: replace with true streaming-ready logic. For now we reuse the
    offline detector on the Muse virtual frontal as a placeholder.
    """

    df_muse = prepared.muse_df_prepped.copy()
    if "pred_stage" in df_muse.columns and "sleep_stage" not in df_muse.columns:
        df_muse = df_muse.rename(columns={"pred_stage": "sleep_stage"})

    if "sleep_stage" not in df_muse.columns:
        df_muse["sleep_stage"] = "N2"

    events, summary = detect_so_events(
        df_muse,
        fs=prepared.fs,
        signal_col="v_frontal_filt",
        sigma_carrier_col="v_frontal",
        preset=preset,
        stages=None,
    )

    print('##########################################')
    print("Online (placeholder) detection summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    _save_events(events, f"so_detection_online_{prepared.sid}.csv")
    return events, summary


def summarize_detection_overlap(
    offline_events: pd.DataFrame,
    online_events: pd.DataFrame,
    *,
    tolerance_s: float = 0.5,
) -> dict:
    """Compare offline and online detections by negative-peak timing."""

    def _extract_times(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if df.empty:
            return np.array([], dtype=float), np.array([], dtype=int)
        col = "t_neg_s" if "t_neg_s" in df.columns else df.columns[0]
        times = df[col].to_numpy(dtype=float)
        positions = np.arange(len(df), dtype=int)
        return times, positions

    offline_times, offline_pos = _extract_times(offline_events)
    online_times, online_pos = _extract_times(online_events)

    matched_pairs: list[dict[str, float | int]] = []
    matched_offline = np.zeros(len(offline_times), dtype=bool)
    matched_online = np.zeros(len(online_times), dtype=bool)

    if offline_times.size and online_times.size:
        order = np.argsort(online_times)
        online_sorted = online_times[order]

        for off_idx, time_val in sorted(enumerate(offline_times), key=lambda item: item[1]):
            if not np.isfinite(time_val):
                continue
            idx = np.searchsorted(online_sorted, time_val)
            candidate_indices: list[int] = []
            if 0 <= idx < online_sorted.size:
                candidate_indices.append(idx)
            if idx - 1 >= 0:
                candidate_indices.append(idx - 1)
            if idx + 1 < online_sorted.size:
                candidate_indices.append(idx + 1)

            best_sorted_idx = None
            best_diff = float("inf")
            for cand in candidate_indices:
                orig_idx = order[cand]
                if matched_online[orig_idx]:
                    continue
                diff = abs(online_sorted[cand] - time_val)
                if diff < best_diff:
                    best_diff = diff
                    best_sorted_idx = cand

            if best_sorted_idx is None or best_diff > tolerance_s:
                continue

            orig_idx = order[best_sorted_idx]
            matched_online[orig_idx] = True
            matched_offline[off_idx] = True
            matched_pairs.append(
                {
                    "offline_idx": int(offline_pos[off_idx]),
                    "online_idx": int(online_pos[orig_idx]),
                    "offline_time": float(offline_times[off_idx]),
                    "online_time": float(online_times[orig_idx]),
                    "time_diff": float(best_diff),
                }
            )

    matched_count = len(matched_pairs)
    offline_only_indices = [int(offline_pos[i]) for i, val in enumerate(matched_offline) if not val]
    online_only_indices = [int(online_pos[i]) for i, val in enumerate(matched_online) if not val]

    results = {
        "offline_events": int(offline_times.size),
        "online_events": int(online_times.size),
        "matched_events": int(matched_count),
        "match_fraction_offline": (matched_count / offline_times.size) if offline_times.size else np.nan,
        "match_fraction_online": (matched_count / online_times.size) if online_times.size else np.nan,
        "tolerance_s": tolerance_s,
        "matched_pairs": matched_pairs,
        "offline_only_indices": offline_only_indices,
        "online_only_indices": online_only_indices,
    }

    print('#########################################')
    print("Offline vs. online detection overlap:")
    for key, value in results.items():
        if key in {"matched_pairs", "offline_only_indices", "online_only_indices"}:
            continue
        print(f"  {key}: {value}")

    return results


def plot_sample_events(
    prepared: PreparedSignals,
    offline_events: pd.DataFrame,
    online_events: pd.DataFrame,
    overlap_info: dict,
    *,
    window_sec: float = 7.0,
    max_examples: int = 5,
) -> None:
    """Plot sample windows for matched and offline-only detections."""

    fs = prepared.fs
    psg = prepared.v_psg_filt
    muse = prepared.v_muse_filt
    total_samples = len(psg)

    def _event_time(df: pd.DataFrame, idx: int) -> float:
        row = df.iloc[int(idx)]
        for col in ("t_neg_s", "t0_s", "t_center_s"):
            if col in df.columns and pd.notna(row.get(col)):
                return float(row[col])
        first_col = df.columns[0]
        return float(row[first_col])

    def _plot_group(entries, title: str, include_online: bool) -> None:
        if not entries:
            print(f"No events to plot for '{title}'.")
            return

        count = min(len(entries), max_examples)
        fig, axes = plt.subplots(count, 1, figsize=(10, 2.4 * count), sharex=False)
        if count == 1:
            axes = [axes]

        # entries_selection = entries[:count]
        entries_selection = random.sample(entries, count)
        for ax, entry in zip(axes, entries_selection):
            if include_online:
                off_idx = entry["offline_idx"]
                on_idx = entry["online_idx"]
                on_time = _event_time(online_events, on_idx)
            else:
                off_idx = entry
                on_idx = None
                on_time = None

            center_time = _event_time(offline_events, off_idx)
            start_time = center_time - window_sec / 2.0
            end_time = center_time + window_sec / 2.0
            start_idx = max(0, int(np.floor(start_time * fs)))
            end_idx = min(total_samples, int(np.ceil(end_time * fs)))
            if end_idx <= start_idx:
                continue

            segment_time = (np.arange(start_idx, end_idx) / fs) - center_time
            ax.plot(segment_time, psg[start_idx:end_idx], color="black", linewidth=1.0, label="PSG virtual (delta)")
            ax.plot(segment_time, muse[start_idx:end_idx], color="red", linewidth=1.0, label="Muse virtual (delta)")
            ax.axvline(0.0, color="blue", linewidth=1.0, linestyle="-", label="Offline neg peak")

            if include_online and on_time is not None:
                offset = on_time - center_time
                ax.axvline(offset, color="green", linewidth=1.0, linestyle="--", label="Online neg peak")
                ax.set_title(
                    f"Matched event (offline idx {off_idx}, online idx {on_idx}, Δ={offset * 1000.0:.1f} ms)"
                )
            else:
                ax.set_title(f"Offline-only event (idx {off_idx})")

            ax.set_xlim(-window_sec / 2.0, window_sec / 2.0)
            ax.set_ylabel("µV")
            ax.grid(True, linewidth=0.3, alpha=0.4)
            ax.legend(loc="upper right", fontsize=8)

        axes[-1].set_xlabel("Time relative to offline neg peak (s)")
        fig.suptitle(title, fontsize=14)
        fig.tight_layout()
        plt.show()

    matched_pairs = overlap_info.get("matched_pairs", [])
    offline_only = overlap_info.get("offline_only_indices", [])

    _plot_group(matched_pairs, "Matched offline/online slow waves", include_online=True)
    _plot_group(offline_only, "Offline detections missed online", include_online=False)


if __name__ == "__main__":
    subject_id = "healthy_older_01"
    data_dir = DEFAULT_DATA_DIR

    prepared_signals = prepare_virtual_signals(subject_id, data_dir, verbose=True)

    compare_muse_to_psg(prepared_signals, plot_window_sec=30.0, verbose=True)

    offline_events, offline_summary = detect_slow_waves_offline(prepared_signals, preset="ngo2013")
    online_events, online_summary = detect_slow_waves_online(prepared_signals, preset="ngo2013")

    overlap_info = summarize_detection_overlap(offline_events, online_events, tolerance_s=1.5)
    plot_sample_events(prepared_signals, offline_events, online_events, overlap_info, window_sec=7.0, max_examples=5)

    evaluate_targeting(online_events, prepared_signals.muse_df_prepped, online_summary, fs=prepared_signals.fs)
