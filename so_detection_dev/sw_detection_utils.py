import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


"""For plotting functions, we use the following data format of stored Muse data:
A pandas DataFrame with columns:
    'ts'   : timestamps (float, seconds since epoch)
    'ch1'  : EEG channel 1 (TP9)
    'ch2'  : EEG channel 2 (AF7)
    'ch3'  : EEG channel 3 (AF8)
    'ch4'  : EEG channel 4 (TP10)
EEG data shape: (5295096, 7). Duration: 344.7 minutes. 
             ts         ch1         ch2         ch3         ch4
0  1.679614e+09  788.534790  790.146545  801.025635  802.637390
1  1.679614e+09  772.014648  788.131897  793.369934  687.399292
"""

ch_names = {"ch1": "TP9", "ch2": "AF7", "ch3": "AF8", "ch4": "TP10"}


def plot_eeg(
    df: pd.DataFrame,
    *,
    channels=("ch1","ch2","ch3","ch4"),
    fs: float = 256.0,            # sampling rate in Hz
    title: str | None = None,
    max_points: int = 250_000,    # decimate for speed if needed
    offset_scale: float = 3.0,    # vertical spacing multiplier (per-channel MAD)
    linewidth: float = 0.8,
    alpha: float = 0.9,
    facecolor: str = "white",
    grid: bool = False,
    y_lim: tuple[float, float] | None = None,  # e.g., (-200, 200) or None for auto
):
    """
    Stacked-overlap plot of EEG channels on a single axis, using the DataFrame
    index as time (constant fs).
    """

    # ---- Validate channels
    ch_present = [c for c in channels if c in df.columns]
    if not ch_present:
        raise ValueError("No EEG channel columns found among: " + ", ".join(channels))

    # ---- Copy & decimate (stride)
    data = df[ch_present].copy()
    N = len(data)
    step = max(1, N // max_points)
    if step > 1:
        data = data.iloc[::step].reset_index(drop=True)

    # ---- Time vector from index & fs
    t = np.arange(len(data)) / float(fs)

    # ---- Colors (Bauhaus-ish: blue, red, yellow, black, green, orange)
    palette = ["#1f77b4", "#d62728", "#ffcc00", "#000000", "#2ca02c", "#ff7f0e"]

    # ---- Robust centers & per-channel offsets
    def mad(x: np.ndarray) -> float:
        med = np.nanmedian(x)
        return np.nanmedian(np.abs(x - med))

    meds = {c: np.nanmedian(data[c].values) for c in ch_present}
    mads = {
        c: (mad(data[c].values) or (np.nanstd(data[c].values) * 0.5) or 1e-9)
        for c in ch_present
    }
    offsets = {c: i * offset_scale * mads[c] for i, c in enumerate(ch_present)}

    # ---- Plot
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained", facecolor=facecolor)
    ax.set_facecolor(facecolor)

    for i, c in enumerate(ch_present):
        y = data[c].values - meds[c] - offsets[c]
        ax.plot(
            t, y,
            lw=linewidth,
            alpha=alpha,
            color=palette[i % len(palette)],
            label=ch_names.get(c, c).upper(),
            solid_joinstyle="round",
            solid_capstyle="round",
            antialiased=True,
        )

    # ---- Aesthetics
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("EEG (µV, robust-centered; vertically offset)")
    if grid:
        ax.grid(True, linewidth=0.4, alpha=0.3)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", which="both", labelsize=10, length=4)

    # Right-edge channel labels at each baseline
    x_end = t[-1] if len(t) else 0.0
    for i, c in enumerate(ch_present):
        ax.text(
            x_end, -offsets[c],
            f" {ch_names.get(c, c).upper()} ",
            va="center", ha="left",
            fontsize=10, color=palette[i % len(palette)],
        )

    # Legend is optional; labels on the right usually suffice
    ax.legend(loc="upper right", frameon=False, fontsize=9, ncols=min(3, len(ch_present)))

    if y_lim is not None:
        ax.set_ylim(y_lim)

    if title:
        fig.suptitle(title, fontsize=14, weight="bold")

    return fig, ax


def plot_eeg_subplots(
    df: pd.DataFrame,
    *,
    channels=("ch1","ch2","ch3","ch4"),
    fs: float = 256.0,   # Hz, default Muse EEG rate
    title: str | None = None,
    max_points: int = 250_000,
    linewidth: float = 0.8,
    alpha: float = 0.9,
    facecolor: str = "white",
    grid: bool = False,
    event = None,
):
    """
    Plot EEG channels in separate subplots (one per channel), using the DataFrame index
    as implicit sampling (constant fs).
    """

    # ---- Validate columns
    ch_present = [c for c in channels if c in df.columns]
    if len(ch_present) == 0:
        raise ValueError("No EEG channel columns found among: " + ", ".join(channels))

    # ---- Copy and decimate if needed
    data = df[ch_present].copy()
    N = len(data)
    step = max(1, N // max_points)
    data = data.iloc[::step].reset_index(drop=True)

    # ---- Time vector from index
    t = np.arange(len(data)) / fs

    # ---- Colors
    palette = ["#1f77b4", "#d62728", "#ffcc00", "#2ca02c", "#000000", "#ff7f0e"]

    # ---- Subplots
    fig, axes = plt.subplots(len(ch_present), 1, figsize=(8, 6), sharex=True)
    if len(ch_present) == 1:
        axes = [axes]

    for i, c in enumerate(ch_present):
        ax = axes[i]
        ax.plot(
            t,
            data[c].values,
            lw=linewidth,
            alpha=alpha,
            color=palette[i % len(palette)],
        )
        ax.set_ylabel(f"{ch_names.get(c, c).upper()} (µV)")
        ax.set_facecolor(facecolor)
        if grid:
            ax.grid(True, linewidth=0.4, alpha=0.3)

        # Simplify look
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        if event is not None:
            # vertical line at 5 in all subplots:
            event_start = 5
            ax.axvline(x=event_start, color='red', linestyle='--', linewidth=1)
            duration_end = event.Duration
            event_end = event_start + event.Duration
            ax.axvline(x=event_end, color='red', linestyle='--', linewidth=1)

    axes[-1].set_xlabel("Time (s)")

    if title:
        fig.suptitle(title, fontsize=14, weight="bold")

    fig.tight_layout()
    return fig, axes




# ------------------------------------------------------------------------------

# Full offline SO (slow oscillation) analysis toolkit in the style of Ngo et al. (2013)
# - Virtual frontal from temporals: see virtual_frontal.make_virtual_frontal_from_temporals
# - Plotting helper to compare raw vs virtual
# - SO detection (stage-selectable; optional downsampling; Ngo-style duration & amplitude gates)
# - Subject-specific up-state delay estimation (median neg->pos latency) and use for sigma-lock window
# - Spindle RMS envelopes (9–12 Hz & 12–15 Hz), baseline-corrected peaks around the estimated up-state
#
# USAGE EXAMPLE (df_eeg with v1, v2, v_frontal, mask_good, v_frontal_filt):
#
# events, summary = detect_so_events(
#     df_eeg, fs=256, signal_col='v_frontal_filt',
#     stages=('N2','N3'),                # default N2+N3; switch to ('N3',) for N3-only
#     preset='ngo2013',                  # 'ngo2013'|'strict'|'lenient'
#     upstate_delay_mode='auto',         # use subject-specific estimate
#     manual_upstate_delay_s=None,       # or set a float (e.g., 0.50) to override
#     sigma_carrier_col='v_frontal',     # channel to compute sigma RMS from
# )
# summary
#
# plot_so_events(df_eeg, events, fs=256, signal_col='v_frontal_filt', start_s=0, end_s=30)
#
# A/B:
# run_ab_detection(df_eeg, fs=256, signal_col='v_frontal_filt', stages=('N2','N3'))
#
# ------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, resample_poly

# # ---------- Filters ----------

def _butter_design(fs, btype, cutoff, order):
    ny = 0.5 * fs
    if btype in ('high', 'low'):
        Wn = cutoff / ny
    elif btype == 'band':
        Wn = [c/ny for c in cutoff]
    else:
        raise ValueError("btype must be 'high'/'low'/'band'")
    return butter(order, Wn, btype=btype)

def hp_filter(x, fs, fc=0.1, order=3):
    b,a = _butter_design(fs, 'high', fc, order)
    return filtfilt(b, a, x)

def lp_filter(x, fs, fc=3.5, order=3):
    b,a = _butter_design(fs, 'low', fc, order)
    return filtfilt(b, a, x)

def bandpass(x, fs, f_lo, f_hi, order=3):
    b,a = _butter_design(fs, 'band', (f_lo, f_hi), order)
    return filtfilt(b, a, x)



def rms_envelope(x, fs, band=(12,15), win_sec=0.20):
    """Band-pass then RMS with sliding window (non-causal, centered)."""
    xb = bandpass(x, fs, band[0], band[1], order=4)
    sq = xb**2
    win = max(1, int(round(win_sec * fs)))
    kernel = np.ones(win, dtype=float) / win
    env = np.sqrt(np.convolve(sq, kernel, mode='same'))
    return env


def hypnogram_to_numeric(stage_series):
    """Convert sleep stage strings to numeric codes for plotting."""
    mapping = {
        'W': 5,
        'N1': 3,
        'N2': 2,
        'N3': 1,
        'R': 4
    }
    return stage_series.map(mapping).fillna(-1).astype(int)

# ----------------------------- Plotting ------------------------------
def plot_virtual_channel_comparison(df_eeg, fs=256, start_idx=0, window_sec=30):
    """
    Compare original Muse channels vs virtual frontal channels.
    """
    n_samples = window_sec * fs
    end_idx = start_idx + n_samples

    # relative time axis
    time = (df_eeg['ts'].iloc[start_idx:end_idx].values - df_eeg['ts'].iloc[start_idx])

    fig, axes = plt.subplots(6, 1, figsize=(10, 10), sharex=True)

    # Raw channels TP9 and TP10 (original)
    axes[0].plot(time, df_eeg['ch1'].iloc[start_idx:end_idx], label="TP9-FPz (raw)", alpha=0.7)
    axes[0].plot(time, df_eeg['ch4'].iloc[start_idx:end_idx], label="TP10-FPz (raw)", alpha=0.7)
    axes[0].legend(loc="upper right")
    axes[0].set_ylabel("µV")

    # noise in channels TP9 and TP10
    i_axis = 1
    axes[i_axis].plot(time, df_eeg['noise_level_channel_1'].iloc[start_idx:end_idx], label="Noise TP9", color="tab:blue")
    axes[i_axis].plot(time, df_eeg['noise_level_channel_4'].iloc[start_idx:end_idx], label="Noise TP10", color="tab:orange")
    axes[i_axis].legend(loc="upper right")
    axes[i_axis].set_ylabel("Noise level")
    
    # v1, v2 (Fpz-TPx style)
    i_axis = 2
    axes[i_axis].plot(time, df_eeg['v1'].iloc[start_idx:end_idx], label="v1 = Fpz-TP9 ([0.1 - 20 Hz])", color="tab:blue")
    axes[i_axis].plot(time, df_eeg['v2'].iloc[start_idx:end_idx], label="v2 = Fpz-TP10 ([0.1 - 20 Hz])", color="tab:orange")
    axes[i_axis].legend(loc="upper right")
    axes[i_axis].set_ylabel("µV")

    # Weighted frontal channel
    i_axis = 3
    axes[i_axis].plot(time, df_eeg['v_frontal'].iloc[start_idx:end_idx], label="v_frontal (weighted)", color="tab:green")
    axes[i_axis].legend(loc="upper right")
    axes[i_axis].set_ylabel("µV")

    # Good mask (binary indicator)
    i_axis = 4
    axes[i_axis].plot(time, df_eeg['mask_good'].iloc[start_idx:end_idx].astype(int), label="mask_good", color="tab:red")
    axes[i_axis].legend(loc="upper right")
    axes[i_axis].set_ylabel("Good?")

    # Filtered frontal channel (SO-band)
    i_axis = 5
    axes[i_axis].plot(time, df_eeg['v_frontal_filt'].iloc[start_idx:end_idx], label="v_frontal_filt (0.1-3.5 Hz)", color="tab:purple")
    axes[i_axis].legend(loc="upper right")
    axes[i_axis].set_ylabel("µV")
    axes[i_axis].set_xlabel("Time (s)")

    sleep_stage = df_eeg['sleep_stage'].iloc[start_idx:end_idx].unique()

    plt.suptitle(f"Comparison of Original vs Virtual Channels ({window_sec} s snippet). Stage(s): {sleep_stage}", fontsize=14)
    plt.tight_layout()
    plt.show()

# --------------------------- SO Detection ----------------------------
def _zero_crossings_pos_to_neg(x):
    """Indices i where x[i-1] >= 0 and x[i] < 0 (discrete)."""
    x_prev = x[:-1]
    x_next = x[1:]
    return np.where((x_prev >= 0) & (x_next < 0))[0] + 1

def _subsample_crossing_time(x, i):
    """
    Estimate the time (in sample units) of the zero crossing between samples i-1 and i,
    where x[i-1] >= 0 and x[i] < 0 (a pos→neg crossing), using linear interpolation.
    Returns a float index (can be between i-1 and i).
    """
    x0, x1 = x[i-1], x[i]
    if x1 == x0:
        return i - 0.5  # degenerate; split the difference
    # Fraction of the way from sample i-1 to i where the line hits zero:
    # x(t) = x0 + (t)*(x1 - x0); solve x(t_cross)=0 => t_cross = x0 / (x0 - x1)
    frac = x0 / (x0 - x1)
    return (i - 1) + frac


def _interval_features(x, i0, i1, fs):
    """Features between successive pos->neg zero-crossings: neg/pos peaks, amp_pp, dur, slope."""
    segment = x[i0:i1]
    if segment.size == 0:
        return None
    rel_min = np.argmin(segment)
    rel_max = np.argmax(segment)
    t_min = i0 + rel_min
    t_max = i0 + rel_max
    neg_peak = segment[rel_min]
    pos_peak = segment[rel_max]
    amp_pp = pos_peak - neg_peak

    # get sub-sample-precision zero-crossing times, for duration and slope
    t0 = _subsample_crossing_time(x, i0)
    t1 = _subsample_crossing_time(x, i1)
    dur = (t1 - t0) / fs
    # Up-slope from negative peak to next zero-crossing
    slope = abs(neg_peak) / max(1e-9, (t1 - t_min) / fs)

    return {
        'i0': i0, 'i1': i1,
        't0_s': t0 / fs, 't1_s': t1 / fs,
        't_neg_s': t_min / fs, 't_pos_s': t_max / fs,
        'neg_peak_uv': float(neg_peak), 'pos_peak_uv': float(pos_peak),
        'amp_pp_uv': float(amp_pp), 'dur_s': float(dur),
        'slope_uv_per_s': float(slope)
    }

def _coverage(mask, i0, i1):
    """Fraction of samples True in [i0, i1)."""
    if mask is None:
        return 1.0
    seg = mask[i0:i1]
    if seg.size == 0:
        return 0.0
    return float(np.mean(seg))

def _stage_mask_from_strings(stage_series, stages=('N2','N3')):
    if stage_series is None:
        return None
    s = stage_series.astype('object').fillna('NaN').values
    allowed = set(stages)
    return np.array([(val in allowed) for val in s], dtype=bool)

PRESETS = {
    'ngo2013': {
        'dur_min': 0.9, 'dur_max': 2.0,
        'neg_mult': 1.25, 'pp_mult': 1.25,
        'min_good_cov': 0.8,    # >=80% usable samples within cycle
        'min_stage_cov': 0.8    # >=80% of cycle in selected stages
    },
    'strict': {
        'dur_min': 0.9, 'dur_max': 2.0,
        'neg_mult': 1.5, 'pp_mult': 1.5,
        'min_good_cov': 0.9,
        'min_stage_cov': 0.9
    },
    'lenient': {
        'dur_min': 0.8, 'dur_max': 2.2,
        'neg_mult': 1.1, 'pp_mult': 1.1,
        'min_good_cov': 0.7,
        'min_stage_cov': 0.7
    }
}

def _estimate_upstate_delay(cycles_df, robust=True):
    """
    Estimate subject-specific up-state delay as the latency from negative peak to positive peak.
    Uses all duration/coverage-passing cycles before amplitude gating for robustness.
    Returns (delay_s, n_used).
    """
    if cycles_df is None or len(cycles_df) == 0:
        return np.nan, 0
    lat = (cycles_df['t_pos_s'] - cycles_df['t_neg_s']).to_numpy()
    lat = lat[np.isfinite(lat) & (lat > 0) & (lat < 2.0)]
    if lat.size == 0:
        return np.nan, 0
    return (np.median(lat) if robust else np.mean(lat)), int(lat.size)

def detect_so_events(df_eeg,
                     fs=256.0,
                     signal_col='v_frontal_filt',
                     stages=('N2','N3'),
                     target_fs=None,
                     preset='ngo2013',
                     spindle_rms=True,
                     spindle_bands=((9,12),(12,15)),
                     spindle_win=0.20,
                     sigma_carrier_col='v_frontal',
                     upstate_delay_mode='auto',       # 'auto' | 'manual' | 'fixed_default'
                     manual_upstate_delay_s=None,      # if upstate_delay_mode == 'manual'
                     default_up_lo_hi=(-0.15, 0.35)):  # window relative to estimated delay (sec)
    """
    Returns:
      events_df: DataFrame of detected SO events with features and sigma-lock peaks.
      summary: dict with counts, distributions, train stats, up-state delay, and preset info.
    """
    params = PRESETS[preset].copy()

    x = df_eeg[signal_col].fillna(method='ffill').fillna(method='bfill').to_numpy(dtype=float)
    n = len(x)

    good_mask = df_eeg['mask_good'].to_numpy(dtype=bool) if 'mask_good' in df_eeg.columns else np.ones(n, dtype=bool)

    stage_mask = _stage_mask_from_strings(df_eeg['sleep_stage'], stages=stages) \
                 if ('sleep_stage' in df_eeg.columns and stages is not None) else None

   
    # Candidate cycles (duration + coverage only)
    zc = _zero_crossings_pos_to_neg(x)
    cand_cycles = []
    for k in range(len(zc)-1):
        
        """
        each window is [pos→neg_k, pos→neg_{k+1}). That interval contains:
        1.	a negative half-wave (down-state) with its negative peak,
        2.	a negative→positive crossing somewhere in the middle,
        3.	a positive half-wave with its positive peak, and finally
        4.	the next pos→neg crossing that closes the cycle.
        """
        
        i0, i1 = int(zc[k]), int(zc[k+1])
        dur = (i1 - i0) / fs
        if not (params['dur_min'] <= dur <= params['dur_max']):
            continue
        good_cov = _coverage(good_mask, i0, i1)
        if good_cov < params['min_good_cov']:
            continue
        stage_cov = _coverage(stage_mask, i0, i1) if stage_mask is not None else 1.0
        if stage_cov < params['min_stage_cov']:
            continue
        feat = _interval_features(x, i0, i1, fs)
        if feat is not None:
            feat['good_cov'] = good_cov
            feat['stage_cov'] = stage_cov
            cand_cycles.append(feat)

    cand_df = pd.DataFrame(cand_cycles) if len(cand_cycles) else pd.DataFrame(columns=[
        't0_s','t1_s','t_neg_s','t_pos_s','neg_peak_uv','pos_peak_uv',
        'amp_pp_uv','dur_s','slope_uv_per_s','good_cov','stage_cov'
    ])

    # Subject-specific up-state delay estimation
    if upstate_delay_mode == 'auto':
        up_delay_s, n_lat = _estimate_upstate_delay(cand_df, robust=True)
        if not np.isfinite(up_delay_s):
            up_delay_s, n_lat = 0.50, 0  # fallback to typical ~0.5 s
    elif upstate_delay_mode == 'manual':
        up_delay_s = float(manual_upstate_delay_s) if manual_upstate_delay_s is not None else 0.50
        n_lat = -1
    else:  # 'fixed_default'
        up_delay_s, n_lat = 0.50, -1

    # Amplitude gating on candidates
    if len(cand_df) == 0:
        events = cand_df.copy()
    else:
        """
        Ngo et al:
        "For SO detection, the EEG was low-pass filtered at 3.5 Hz, downsampled to 100 Hz, and positive-to-negative zero 
        crossings were determined. Intervals between 0.9 and 2.0 s were classified as putative SOs. Only those intervals 
        were selected whose trough-to-peak amplitude exceeded 1.25× the average amplitude and whose troughs were more 
        negative than 1.25× the average trough.”
        I.e., this is biasing towards larger events, which may be desirable (Ngo et al likely strongest 40-50% of events).
        """
        
        mean_neg = cand_df['neg_peak_uv'].mean()
        mean_pp  = cand_df['amp_pp_uv'].mean()
        neg_thr = params['neg_mult'] * mean_neg
        pp_thr  = params['pp_mult'] * mean_pp
        keep = (cand_df['neg_peak_uv'] < neg_thr) & (cand_df['amp_pp_uv'] > pp_thr)
        events = cand_df.loc[keep].reset_index(drop=True)

        # add another estimation of up-state delay from the final events
        if upstate_delay_mode == 'auto' and len(events):
            up_delay_filtered_s, n_filtered_lat = _estimate_upstate_delay(events, robust=True)
            if not np.isfinite(up_delay_filtered_s):
                up_delay_filtered_s, n_filtered_lat = 0.50, 0  # fallback to typical ~0.5 s
        else:
            up_delay_filtered_s, n_filtered_lat = up_delay_s, n_lat
            
        
    # Train grouping (IPI <= 2s from negative peaks)
    if len(events):
        tneg = events['t_neg_s'].values
        ipi = np.diff(tneg) if len(tneg) > 1 else np.array([])
        same_train = np.concatenate([[False], ipi <= 2.0]) if ipi.size else np.array([False])
        train_id = np.zeros(len(tneg), dtype=int)
        tid = 0
        for i in range(1, len(tneg)):
            if same_train[i]:
                train_id[i] = tid
            else:
                tid += 1
                train_id[i] = tid
        events['train_id'] = train_id
    else:
        events['train_id'] = []

    # Spindle-lock metrics using subject-specific up-state delay
    spin_metrics = {}
    if spindle_rms and len(events):
        if sigma_carrier_col in df_eeg.columns:
            carrier = df_eeg[sigma_carrier_col].fillna(method='ffill').fillna(method='bfill').to_numpy(dtype=float)
        else:
            # fallback to the detection signal (pre-filtered)
            carrier = df_eeg[signal_col].fillna(method='ffill').fillna(method='bfill').to_numpy(dtype=float)

        slow_env = rms_envelope(carrier, fs, band=tuple(spindle_bands[0]), win_sec=spindle_win)
        fast_env = rms_envelope(carrier, fs, band=tuple(spindle_bands[1]), win_sec=spindle_win)

        def idx_window(abs_t, w, fs_local, N):
            i0 = int(round(abs_t * fs_local))
            j0 = i0 + int(round(w[0] * fs_local))
            j1 = i0 + int(round(w[1] * fs_local))
            j0 = max(0, min(N, j0)); j1 = max(0, min(N, j1))
            return j0, j1

        slow_peaks, fast_peaks = [], []
        for _, row in events.iterrows():
            t0 = float(row['t_neg_s'])
            # baseline window relative to negative peak
            b0, b1 = idx_window(t0, (-1.0, -0.5), fs, len(carrier))
            # up-state window relative to NEGATIVE peak, but centered on estimated up-state delay
            u0, u1 = idx_window(t0 + up_delay_s, default_up_lo_hi, fs, len(carrier))

            base_slow = np.nanmean(slow_env[b0:b1]) if (b1 > b0) else 0.0
            base_fast = np.nanmean(fast_env[b0:b1]) if (b1 > b0) else 0.0
            peak_slow = np.nanmax(slow_env[u0:u1]) if (u1 > u0) else np.nan
            peak_fast = np.nanmax(fast_env[u0:u1]) if (u1 > u0) else np.nan
            slow_peaks.append(peak_slow - base_slow)
            fast_peaks.append(peak_fast - base_fast)

        events['slow_sigma_rms_peak'] = slow_peaks
        events['fast_sigma_rms_peak'] = fast_peaks

        spin_metrics = {
            'slow_sigma_mean_peak': float(np.nanmean(slow_peaks)) if len(slow_peaks) else np.nan,
            'slow_sigma_std_peak': float(np.nanstd(slow_peaks)) if len(slow_peaks) else np.nan,
            'fast_sigma_mean_peak': float(np.nanmean(fast_peaks)) if len(fast_peaks) else np.nan,
            'fast_sigma_std_peak': float(np.nanstd(fast_peaks)) if len(fast_peaks) else np.nan
        }

    # Summary
    train_sizes = events.groupby('train_id').size().values if len(events) else np.array([])
    summary = {
        'count': int(len(events)),
        'preset': preset,
        'stages': tuple(stages) if stages is not None else None,
        'mean_neg_peak_uv': float(events['neg_peak_uv'].mean()) if len(events) else np.nan,
        'mean_amp_pp_uv': float(events['amp_pp_uv'].mean()) if len(events) else np.nan,
        'mean_dur_s': float(events['dur_s'].mean()) if len(events) else np.nan,
        'mean_slope': float(events['slope_uv_per_s'].mean()) if len(events) else np.nan,
        'train_len_mean': float(np.mean(train_sizes)) if train_sizes.size else np.nan,
        'train_len_median': float(np.median(train_sizes)) if train_sizes.size else np.nan,
        'n_trains_ge2': int(np.sum(train_sizes >= 2)) if train_sizes.size else 0,
        'n_trains_ge3': int(np.sum(train_sizes >= 3)) if train_sizes.size else 0,
        'upstate_delay_s': float(up_delay_s),
        'upstate_delay_n': int(n_lat),
        'upstate_delay_s_filtered': float(up_delay_filtered_s),
        'upstate_delay_n_filtered': int(n_filtered_lat)
    }
    summary.update(spin_metrics)

    return events, summary

def run_ab_detection(df_eeg, fs=256.0, signal_col='v_frontal_filt', stages=('N2','N3')):
    """
    Runs detection with three presets (ngo2013/strict/lenient) and returns a summary table.
    Uses subject-specific up-state delay for all presets.
    """
    rows = []
    for preset in ['ngo2013','strict','lenient']:
        events, summ = detect_so_events(
            df_eeg, fs=fs, signal_col=signal_col, stages=stages,
            target_fs=None, preset=preset, spindle_rms=True,
            upstate_delay_mode='auto', manual_upstate_delay_s=None
        )
        rows.append({
            'preset': preset,
            'stages': ','.join(stages),
            'count': summ['count'],
            'mean_amp_pp_uv': summ['mean_amp_pp_uv'],
            'mean_neg_peak_uv': summ['mean_neg_peak_uv'],
            'mean_dur_s': summ['mean_dur_s'],
            'mean_slope': summ['mean_slope'],
            'train_len_mean': summ['train_len_mean'],
            'train_len_median': summ['train_len_median'],
            'n_trains_ge2': summ['n_trains_ge2'],
            'n_trains_ge3': summ['n_trains_ge3'],
            'fast_sigma_mean_peak': summ.get('fast_sigma_mean_peak', np.nan),
            'slow_sigma_mean_peak': summ.get('slow_sigma_mean_peak', np.nan),
            'upstate_delay_s': summ.get('upstate_delay_s', np.nan),
            'upstate_delay_n': summ.get('upstate_delay_n', 0),
            'upstate_delay_s_filtered': summ.get('upstate_delay_s_filtered', np.nan),
            'upstate_delay_n_filtered': summ.get('upstate_delay_n_filtered', 0),
        })
    return pd.DataFrame(rows)


def plot_so_events(df_eeg, events, fs=256.0, signal_col='v_frontal_filt', start_s=None, end_s=None):
    """
    Visualization: SO-band signal with detected negative peaks overlaid.

    Notes
    -----
    - Uses index-derived time (`np.arange(N)/fs`) to avoid relying on `ts`, which may be
      coarse or irregular in some datasets. Event times from detection are in seconds
      relative to start, so this aligns correctly.
    """
    x = df_eeg[signal_col].to_numpy(dtype=float)
    n = len(x)
    t = np.arange(n, dtype=float) / float(fs)

    if start_s is None:
        start_s = 0.0
    if end_s is None:
        end_s = float(t[-1]) if n else 0.0

    mask = (t >= start_s) & (t <= end_s)

    plt.figure(figsize=(14, 4))
    plt.plot(t[mask], x[mask], label="SO-band signal")

    tneg = events['t_neg_s'].values if len(events) else np.array([])
    inwin = (tneg >= start_s) & (tneg <= end_s)
    tneg_in = tneg[inwin]
    # Sample-aligned amplitudes for markers
    idx = np.clip(np.round(tneg_in * fs).astype(int), 0, max(0, n - 1))
    yneg = x[idx] if n else np.array([])

    plt.plot(tneg_in, yneg, linestyle='none', marker='o', label="SO negative peaks")
    plt.xlabel("Time (s)")
    plt.ylabel("µV")
    plt.title("Detected Slow Oscillations")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
    
    
# ---------------- Adaptive delay estimation & assessment ----------------


class AdaptiveDelayEstimator:
    """Online-friendly estimator for neg->pos latency using a rolling buffer."""
    def __init__(self, max_events: int = 80):
        self.max_events = int(max_events)
        self._lat = []
    def update(self, t_neg_s: float, t_pos_s: float):
        d = float(t_pos_s - t_neg_s)
        if not (0 < d < 2.0) or not np.isfinite(d): return
        self._lat.append(d)
        if len(self._lat) > self.max_events:
            self._lat = self._lat[-self.max_events:]
    def current_delay(self, robust: bool = True, default: float = 0.50) -> float:
        if not self._lat: return float(default)
        a = np.array(self._lat, dtype=float)
        a = a[np.isfinite(a) & (a > 0) & (a < 2.0)]
        if a.size == 0: return float(default)
        return float(np.median(a) if robust else np.mean(a))
    
    

# ------------------------- Event Snippet Plotting Utility -------------------------
from math import ceil

def plot_event_snippets(df_eeg: pd.DataFrame,
                        events: pd.DataFrame,
                        n_samples: int,
                        *,
                        signal_col: str = 'v_frontal_filt',
                        fs: float = 256.0,
                        delay_s: float | None = None,  # if given, vertical line at t0 + delay_s
                        mode: str = 'random',           # 'random' or 'consecutive'
                        start_index: int = 0,           # used when mode='consecutive'
                        cols: int = 5,
                        t_before: float = 3.5,
                        t_after: float = 3.5,
                        facecolor: str = 'white',
                        linewidth: float = 0.9,
                        alpha: float = 0.95,
                        title: str | None = None,
                        seed: int | None = 1234):
    """
    Plot 7-second snippets (t_before + t_after = 7.0 s by default) around each selected SO event.
    For each subplot, draw red vertical lines at: start (t0_s), neg peak (t_neg_s), pos peak (t_pos_s), end (t1_s).

    Parameters
    ----------
    df_eeg : DataFrame containing 'ts' (seconds, absolute) and the chosen `signal_col`.
    events : DataFrame from `detect_so_events` with columns t0_s, t_neg_s, t_pos_s, t1_s.
    n_samples : int, number of events to plot.
    signal_col : str, which signal to visualize (default 'v_frontal_filt').
    fs : float, sampling rate (Hz).
    mode : 'random' or 'consecutive'. If 'consecutive', uses `start_index` and the next n events.
    start_index : starting event index for consecutive mode (0-based in the events DataFrame order).
    cols : number of subplot columns (rows computed automatically).
    t_before / t_after : seconds before/after the event *start* (t0_s) to include in each snippet.
    facecolor, linewidth, alpha, title, seed : plot aesthetics.

    Returns
    -------
    fig, axes : the matplotlib Figure and array of Axes.
    """
    if len(events) == 0 or n_samples <= 0:
        raise ValueError("No events to plot or n_samples <= 0.")

    # Select which events to show
    ev = events.reset_index(drop=True)
    if mode not in ("random", "consecutive"):
        raise ValueError("mode must be 'random' or 'consecutive'")

    if mode == 'random':
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(ev), size=min(n_samples, len(ev)), replace=False)
        idx.sort()
        subset = ev.iloc[idx].copy()
    else:  # consecutive
        i0 = int(max(0, start_index))
        i1 = int(min(len(ev), i0 + n_samples))
        subset = ev.iloc[i0:i1].copy()
        if len(subset) == 0:
            raise ValueError("start_index out of range for consecutive selection")

    # Build absolute time vector from df_eeg['ts'] relative to recording start
    if 'ts' not in df_eeg.columns:
        raise ValueError("df_eeg must contain a 'ts' column of absolute seconds")
    t_abs = df_eeg['ts'].to_numpy(dtype=float) - float(df_eeg['ts'].iloc[0])
    x = df_eeg[signal_col].to_numpy(dtype=float)

    # Create subplot grid
    n = len(subset)
    cols = max(1, int(cols))
    rows = int(ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7, 2.6*rows), sharex=False, sharey=False)
    if rows == 1 and cols == 1:
        axes = np.array([[axes]]);
    elif rows == 1:
        axes = np.array([axes]);
    elif cols == 1:
        axes = np.array([[ax] for ax in axes]);

    # Helper for each subplot
    for k, (_, evrow) in enumerate(subset.iterrows()):
        r = k // cols
        c = k % cols
        ax = axes[r, c]
        ax.set_facecolor(facecolor)

        t0 = float(evrow['t0_s']) if 't0_s' in evrow else float(evrow['t_neg_s'])
        t_start = t0 - float(t_before)
        t_end   = t0 + float(t_after)

        m = (t_abs >= t_start) & (t_abs <= t_end)
        if not np.any(m):
            # If the window is out of range (e.g., near edges), skip plotting data but keep markers if inside axes
            ax.text(0.5, 0.5, 'No data in window', transform=ax.transAxes, ha='center', va='center', fontsize=9)
            ax.axis('off')
            continue

        # Plot signal segment
        t_seg = t_abs[m] - t0  # center x-axis around event start (so 0 at t0_s)
        x_seg = x[m]
        ax.plot(t_seg, x_seg, lw=linewidth, alpha=alpha)

        # Vertical markers at event landmarks (relative to t0)
        def _vline_if_in_window(t_marker, label, color='red'):
            if np.isfinite(t_marker) and (t_start <= t_marker <= t_end):
                ax.axvline(x=(t_marker - t0), color=color, linestyle='--', linewidth=1.0)

        _vline_if_in_window(float(evrow.get('t0_s', np.nan)), 'start', 'black')
        _vline_if_in_window(float(evrow.get('t_neg_s', np.nan)), 'neg', 'red')
        _vline_if_in_window(float(evrow.get('t_pos_s', np.nan)), 'pos', 'orange')
        _vline_if_in_window(float(evrow.get('t1_s', np.nan)), 'end', 'black')
        if delay_s is not None and np.isfinite(delay_s):
            _vline_if_in_window(float(evrow.get('t_neg_s', np.nan)) + delay_s, 'est_up', 'green')
            
        ax.set_xlim([-t_before, t_after])
        ax.set_title(f"ev#{int(evrow.name)}  t0={t0:.2f}s", fontsize=9);
        ax.set_xlabel("Time rel. to start (s)", fontsize=8)
        ax.set_ylabel(signal_col, fontsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(labelsize=8, length=3)
        
        ax.axhline(y=0.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
        
    # Hide any unused axes
    total_axes = rows * cols
    for k in range(n, total_axes):
        r = k // cols
        c = k % cols
        axes[r, c].axis('off')

    if title:
        fig.suptitle(title, fontsize=12, weight='bold', y=1)
    fig.tight_layout()
    return fig, axes
