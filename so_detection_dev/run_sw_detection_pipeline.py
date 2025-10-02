"""Run Muse slow-wave detection and targeting assessment with plotting outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sw_detection_utils import detect_so_events, plot_eeg, plot_so_events
from sw_assess_target_performance import (
    assess_targeting_performance,
    assess_targeting_performance_adaptive,
    build_delay_series_ema,
    build_delay_series_kalman,
    plot_targeting_diagnostics,
)
import sys
sys.path.append("..")
from virtual_frontal import make_virtual_frontal_from_temporals, lowpass_delta_frontal


def load_muse_eeg(data_dir: Path, file_name: str | None = None) -> Tuple[pd.DataFrame, Path]:
    """Load a Muse EEG CSV file from the dataset directory."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    candidates = sorted(data_dir.glob("*_eeg.csv"))
    if not candidates:
        raise FileNotFoundError(f"No EEG CSV files found in {data_dir}")

    if file_name is not None:
        target = data_dir / file_name
        if not target.exists():
            raise FileNotFoundError(f"Requested EEG file not found: {target}")
        csv_path = target
    else:
        csv_path = candidates[0]

    df = pd.read_csv(csv_path)
    print(f"Loaded EEG recording: {csv_path.name} (shape={df.shape})")
    return df, csv_path


def ensure_noise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add noise columns if the recording is missing them."""
    df = df.copy()
    for idx in (1, 4):
        col = f"noise_level_channel_{idx}"
        if col not in df:
            df[col] = 0.0
    return df


def prepare_virtual_channels(df: pd.DataFrame, fs: float) -> pd.DataFrame:
    """Augment the EEG DataFrame with virtual frontal channels and masks."""
    df = ensure_noise_columns(df)

    v_frontal, v1, v2, mask_good = make_virtual_frontal_from_temporals(
        df, fs=fs, return_details=True
    )
    df = df.assign(
        v1=v1,
        v2=v2,
        v_frontal=v_frontal,
        mask_good=mask_good.astype(bool),
        v_frontal_filt=lowpass_delta_frontal(v_frontal, fs=fs),
    )
    return df


def run_detection(df: pd.DataFrame, fs: float):
    """Run the slow-wave detector and return events plus summary."""
    events, summary = detect_so_events(df, fs=fs, signal_col="v_frontal_filt")
    print("Detection summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    return events, summary


def kalman_hyperparam_sweep(
    events: pd.DataFrame,
    signal: np.ndarray,
    fs: float,
    *,
    min_alpha: float = 0.02,
    half_life_bounds: tuple[float, float] = (1800.0, 7200.0),
    n_experiments: int = 10,
    betas: tuple[float, ...] = (0.05, 0.10, 0.20),
) -> pd.DataFrame:
    """Run a small sweep of Kalman delay parameters and collect targeting metrics."""

    if len(events) == 0:
        return pd.DataFrame(
            columns=[
                "half_life_s",
                "beta",
                "sec_err_mean_ms",
                "sec_err_median_ms",
                "hit_rate_50ms",
                "hit_rate_100ms",
                "plv",
            ]
        )

    hl_lo, hl_hi = half_life_bounds
    half_life_values = np.linspace(float(hl_lo), float(hl_hi), num=n_experiments, dtype=float)
    beta_values = betas if betas else (0.1,)

    records: list[dict[str, float]] = []
    for idx, half_life_s in enumerate(half_life_values):
        beta = float(beta_values[idx % len(beta_values)])
        delay_series = build_delay_series_kalman(
            events,
            half_life_s=float(half_life_s),
            min_alpha=min_alpha,
            beta=beta,
        )
        if delay_series.empty:
            continue

        summary, _ = assess_targeting_performance_adaptive(
            events,
            signal,
            fs,
            delay_series,
            return_series=False,
        )

        sec_mean = summary.get("sec_err_mean", np.nan)
        sec_median = summary.get("sec_err_median", np.nan)
        records.append(
            {
                "half_life_s": float(half_life_s),
                "beta": beta,
                "sec_err_mean_ms": float(sec_mean * 1000.0) if np.isfinite(sec_mean) else np.nan,
                "sec_err_median_ms": float(sec_median * 1000.0) if np.isfinite(sec_median) else np.nan,
                "hit_rate_50ms": summary.get("sec_hit_rate_50ms", np.nan),
                "hit_rate_100ms": summary.get("sec_hit_rate_100ms", np.nan),
                "plv": summary.get("plv", np.nan),
            }
        )

    return pd.DataFrame.from_records(records)


def evaluate_targeting(events: pd.DataFrame, df: pd.DataFrame, summary: dict, fs: float):
    """Evaluate click targeting performance using fixed and adaptive delays."""
    if events.empty:
        print("No slow-wave events detected; skipping targeting assessment.")
        return

    delay_s = float(summary.get("upstate_delay_s", 0.5))
    fixed_perf, series = assess_targeting_performance(
        events,
        df["v_frontal_filt"].to_numpy(dtype=float),
        fs,
        delay_s,
        return_series=True,
    )
    print("\nFixed-delay targeting metrics:")
    for key, value in fixed_perf.items():
        print(f"  {key}: {value}")

    if series is not None:
        plot_targeting_diagnostics(
            series["sec_err"],
            series["phase_err"],
            fixed_perf,
            title="Fixed-delay targeting diagnostics",
        )

    def _plot_delay_series(delay_df: pd.DataFrame, title: str):
        plt.figure(figsize=(10, 4))
        plt.plot(delay_df["t_center_s"], delay_df["delay_s"], marker="o")
        plt.xlabel("Time since recording start (s)")
        plt.ylabel("Estimated neg→pos delay (s)")
        plt.title(title)
        plt.grid(True, linewidth=0.3, alpha=0.4)
        plt.tight_layout()

    adaptive_configs = [
        ("EMA", build_delay_series_ema, {"half_life_s": 3600.0, "min_alpha": 0.02}, "EMA slow-wave delay estimate"),
        (
            "Kalman",
            build_delay_series_kalman,
            {"half_life_s": 3600.0, "min_alpha": 0.02},
            "Kalman slow-wave delay estimate",
        ),
    ]

    adaptive_signal = df["v_frontal_filt"].to_numpy(dtype=float)

    for label, builder, kwargs, title in adaptive_configs:
        delay_series = builder(events, **kwargs)
        if delay_series.empty:
            print(f"\n{label} adaptive targeting metrics: no delay series (insufficient events)")
            continue

        _plot_delay_series(delay_series, title)

        adaptive_perf, series_adaptive = assess_targeting_performance_adaptive(
            events,
            adaptive_signal,
            fs,
            delay_series,
            return_series=True,
        )

        print(f"\n{label} adaptive targeting metrics:")
        for key, value in adaptive_perf.items():
            print(f"  {key}: {value}")

        if series_adaptive is not None:
            plot_targeting_diagnostics(
                series_adaptive["sec_err"],
                series_adaptive["phase_err"],
                adaptive_perf,
                title=f"{label} adaptive targeting diagnostics",
            )

    sweep_df = kalman_hyperparam_sweep(
        events,
        adaptive_signal,
        fs,
        min_alpha=0.02,
        half_life_bounds=(1800.0, 7200.0),
        n_experiments=10,
        betas=(0.05, 0.10, 0.20),
    )

    if not sweep_df.empty:
        ordered = sweep_df.sort_values(by="sec_err_mean_ms", key=lambda s: np.abs(s))
        print("\nKalman hyperparameter sweep (mean ms error ascending):")
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print(ordered.to_string(index=False, float_format=lambda x: f"{x:6.2f}"))



def plot_overview(df: pd.DataFrame, events: pd.DataFrame, fs: float, start_sec: float = 0.0, window_sec: float = 30.0):
    """Plot key signals and detected events for a short window."""
    start_idx = int(start_sec * fs)
    end_idx = int((start_sec + window_sec) * fs)

    plot_eeg(
        df.iloc[start_idx:end_idx],
        channels=("ch1", "ch4", "v_frontal", "v_frontal_filt"),
        fs=fs,
        title="Muse channels vs virtual frontal signal",
    )

    plot_so_events(
        df,
        events,
        fs=fs,
        signal_col="v_frontal_filt",
        start_s=start_sec,
        end_s=start_sec + window_sec,
    )



def run_pipeline(
    *,
    data_dir: Path,
    fs: float,
    file_name: str | None = None,
    overview_start_sec: float = 0.0,
    overview_window_sec: float = 30.0,
    plot_full_overview: bool = False,
    show_plots: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, Path]:
    """Run the full slow-wave detection pipeline and optionally show plots."""
    
    print(data_dir)
    
    df_raw, csv_path = load_muse_eeg(data_dir=data_dir, file_name=file_name)
    df_prepped = prepare_virtual_channels(df_raw, fs=fs)
    events, summary = run_detection(df_prepped, fs=fs)

    plot_overview(
        df_prepped,
        events,
        fs=fs,
        start_sec=overview_start_sec,
        window_sec=overview_window_sec,
    )

    if plot_full_overview:
        plot_overview(
            df_prepped,
            events,
            fs=fs,
            start_sec=0.0,
            window_sec=df_prepped.shape[0] / fs,
        )

    evaluate_targeting(events, df_prepped, summary, fs=fs)

    if show_plots:
        plt.show()

    return df_prepped, events, summary, csv_path
