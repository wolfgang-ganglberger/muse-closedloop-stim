# Re-run after kernel reset: include all definitions again and self-test.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
from math import sqrt, pi

def _butter_band(fs, f_lo, f_hi, order=4):
    ny = 0.5 * fs
    b, a = butter(order, [f_lo/ny, f_hi/ny], btype='band')
    return b, a

def _band_hilbert_phase_safe(x, fs, f_lo=0.16, f_hi=1.25, order=4):
    """
    NaN-robust: interpolate NaNs (both directions), zero-center, band-pass, Hilbert phase.
    Returns phase array (radians) or an all-NaN array if filtering fails.
    """
    x = np.asarray(x, dtype=float)
    if not np.any(np.isfinite(x)):
        return np.full_like(x, np.nan)
    # interpolate NaNs (linear), then forward/backward fill as needed
    xs = pd.Series(x)
    x_clean = xs.interpolate(limit_direction='both').to_numpy()
    # if still NaNs at edges, replace with 0
    if np.any(~np.isfinite(x_clean)):
        x_clean = np.nan_to_num(x_clean, nan=0.0)
    # detrend (remove median)
    x_clean = x_clean - np.nanmedian(x_clean)
    # band-pass + hilbert
    try:
        b, a = _butter_band(fs, f_lo, f_hi, order=order)
        xf = filtfilt(b, a, x_clean)
        analytic = hilbert(xf)
        phase = np.angle(analytic)
        return phase
    except Exception as e:
        # If filtfilt/hilbert choke, return NaNs of same shape
        return np.full_like(x, np.nan)

def _interp_at_times(x, fs, t_s):
    idx = np.clip(np.round(np.asarray(t_s) * fs).astype(int), 0, len(x)-1)
    return x[idx]

def _circular_mean(theta):
    return np.arctan2(np.sum(np.sin(theta)), np.sum(np.cos(theta)))

def _plv(theta):
    C = np.mean(np.cos(theta))
    S = np.mean(np.sin(theta))
    return np.hypot(C, S)

def _rayleigh_test(theta):
    theta = np.asarray(theta)
    N = theta.size
    if N == 0:
        return np.nan, np.nan
    R = N * _plv(theta)
    Z = (R**2) / N
    p = np.exp(-Z) * (1 + (2*Z - Z**2) / (4*N) - (24*Z - 132*Z**2 + 76*Z**3 - 9*Z**4) / (288*N**2))
    return Z, float(np.clip(p, 0.0, 1.0))

def assess_targeting_performance(events: pd.DataFrame,
                                 signal: np.ndarray,
                                 fs: float,
                                 delay_s: float,
                                 phase_band=(0.16, 1.25),
                                 return_series=False):
    """
    Robust assessment of click timing vs up-state. Uses Hilbert phase; falls back to
    duration-normalized phase if Hilbert fails.
    """
    # seconds error
    t_neg = events['t_neg_s'].to_numpy(dtype=float)
    t_pos = events['t_pos_s'].to_numpy(dtype=float)
    t_click = t_neg + float(delay_s)
    sec_err = t_click - t_pos
    
    # Try Hilbert-phase method
    phi = _band_hilbert_phase_safe(signal, fs, f_lo=phase_band[0], f_hi=phase_band[1], order=4)
    phi_click = _interp_at_times(phi, fs, t_click) if np.any(np.isfinite(phi)) else np.full_like(sec_err, np.nan)
    phi_pos   = _interp_at_times(phi, fs, t_pos)   if np.any(np.isfinite(phi)) else np.full_like(sec_err, np.nan)
    phase_err = np.angle(np.exp(1j*(phi_click - phi_pos)))  # may be all NaN
    
    # If Hilbert failed (all NaNs), use fallback based on each event's cycle duration
    if not np.any(np.isfinite(phase_err)):
        if all(col in events.columns for col in ['t0_s', 't1_s']):
            T = (events['t1_s'] - events['t0_s']).to_numpy(dtype=float)  # cycle length
            # avoid division by zero
            T = np.where(T <= 0, np.nan, T)
            phase_err = 2*np.pi * (sec_err / T)  # radians
            # wrap to [-pi, pi]
            phase_err = (phase_err + np.pi) % (2*np.pi) - np.pi
        else:
            # No way to compute fallback: keep NaNs
            pass
    
    # seconds stats
    def _iqr(x): 
        q75, q25 = np.percentile(x, [75, 25]) if x.size else (np.nan, np.nan)
        return q75 - q25
    def _hit_rate(x, tol):
        return float(np.mean(np.abs(x) <= tol)) if x.size else np.nan
    
    n = sec_err.size
    summary = {
        'n': int(n),
        'sec_err_mean': float(np.mean(sec_err)) if n else np.nan,
        'sec_err_median': float(np.median(sec_err)) if n else np.nan,
        'sec_err_iqr': float(_iqr(sec_err)) if n else np.nan,
        'sec_hit_rate_25ms': _hit_rate(sec_err, 0.025),
        'sec_hit_rate_50ms': _hit_rate(sec_err, 0.050),
        'sec_hit_rate_75ms': _hit_rate(sec_err, 0.075),
        'sec_hit_rate_100ms': _hit_rate(sec_err, 0.100),
        'sec_hit_rate_250ms': _hit_rate(sec_err, 0.250),
    }
    
    # circular stats (only if some finite values exist)
    finite_phi = np.isfinite(phase_err)
    if np.any(finite_phi):
        pe = phase_err[finite_phi]
        plv = _plv(pe)
        Z, p = _rayleigh_test(pe)
        circ_mean = _circular_mean(pe)
        circ_sd = float(np.sqrt(2*(1 - plv))) if np.isfinite(plv) else np.nan
        summary.update({
            'plv': float(plv) if np.isfinite(plv) else np.nan,
            'rayleigh_Z': float(Z) if np.isfinite(Z) else np.nan,
            'rayleigh_p': p,
            'phase_err_mean': float(circ_mean) if np.isfinite(circ_mean) else np.nan,
            'phase_err_sd': circ_sd
        })
    else:
        summary.update({
            'plv': np.nan, 'rayleigh_Z': np.nan, 'rayleigh_p': np.nan,
            'phase_err_mean': np.nan, 'phase_err_sd': np.nan
        })
    
    series = {'sec_err': sec_err, 'phase_err': phase_err, 't_click': t_click, 't_pos': t_pos}
    return (summary, series) if return_series else (summary, None)


# -------- Robustness self-test --------
def _self_test():
    fs = 256.0
    T = 30.0
    t = np.arange(int(T*fs)) / fs
    f0 = 0.9
    so = 60*np.sin(2*np.pi*f0*t)
    x = so.copy()
    # Inject NaN gaps
    x[1000:1200] = np.nan
    x[4000:4200] = np.nan
    # Events: negative peaks roughly every 1/f0
    period = 1.0/f0
    t_neg = np.arange(1.0, T-1.0, period)
    delay_true = period/2.0
    t_pos = t_neg + delay_true
    events = pd.DataFrame({'t_neg_s': t_neg, 't_pos_s': t_pos,
                           't0_s': t_neg - delay_true, 't1_s': t_pos + delay_true})
    delay_est = delay_true - 0.02  # slightly biased
    summary, series = assess_targeting_performance(events, x, fs, delay_est, return_series=True)
    assert summary['n'] == len(events)
    # Ensure we got finite phase stats via Hilbert or fallback
    assert np.isfinite(summary['sec_err_mean'])
    assert np.isfinite(summary['sec_err_median'])
    assert ('plv' in summary)
    return summary
def plot_targeting_diagnostics(sec_err,
                                phase_err,
                                summary,
                                title="Targeting diagnostics",
                                bins=50):
    """Render paired timing-error and phase diagnostics."""

    sec_err = np.asarray(sec_err, dtype=float)
    phase_err = np.asarray(phase_err, dtype=float)

    fig = plt.figure(figsize=(12, 5))
    ax_hist = fig.add_subplot(1, 2, 1)
    ax_polar = fig.add_subplot(1, 2, 2, projection='polar')

    # ---- Timing error histogram (ms)
    err_ms = sec_err * 1000.0
    finite_ms = np.isfinite(err_ms)
    if np.any(finite_ms):
        ax_hist.hist(err_ms[finite_ms], bins=bins, color='tab:gray', alpha=0.75)
    ax_hist.axvline(0.0, color='tab:red', linestyle='--', linewidth=1.2, label='Perfect timing')
    ax_hist.set_xlim(-1000.0, 1000.0)

    if 0:
        # color the 1 SD range
        if np.any(finite_ms):
            std_ms = float(np.std(np.abs(err_ms[finite_ms])))
            ax_hist.fill_betweenx(
                [0, ax_hist.get_ylim()[1]],
                -std_ms,
                std_ms,
                color='tab:orange',
                alpha=0.2,
                label='1 SD range'
            )

    ax_hist.set_xlabel('Timing error (ms)')
    ax_hist.set_ylabel('Count')
    ax_hist.set_title('Timing error distribution')
    ax_hist.legend(loc='upper left', frameon=False)

    mean_ms = summary.get('sec_err_mean') * 1000.0
    median_ms = summary.get('sec_err_median') * 1000.0
    n_events = summary.get('n')
    hit_25 = summary.get('sec_hit_rate_25ms')
    hit_50 = summary.get('sec_hit_rate_50ms')
    hit_100 = summary.get('sec_hit_rate_100ms')
    hit_250 = summary.get('sec_hit_rate_250ms')

    info_lines = []
    if n_events is not None:
        info_lines.append(f'n events: {n_events}')
    if np.isfinite(mean_ms):
        info_lines.append(f'mean error: {mean_ms:.1f} ms')
    if np.isfinite(median_ms):
        info_lines.append(f'median error: {median_ms:.1f} ms')
    if hit_25 is not None and np.isfinite(hit_25):
        info_lines.append(f'hit ≤25 ms: {hit_25*100:5.1f}%')
    if hit_50 is not None and np.isfinite(hit_50):
        info_lines.append(f'hit ≤50 ms: {hit_50*100:5.1f}%')
    if hit_100 is not None and np.isfinite(hit_100):
        info_lines.append(f'hit ≤100 ms: {hit_100*100:5.1f}%')
    if hit_250 is not None and np.isfinite(hit_250):
        info_lines.append(f'hit ≤250 ms: {hit_250*100:5.1f}%')

    if info_lines:
        ax_hist.text(
            0.98,
            0.95,
            '\n'.join(info_lines),
            transform=ax_hist.transAxes,
            ha='right',
            va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, linewidth=0.0)
        )

    # ---- Phase polar plot
    finite_phase = phase_err[np.isfinite(phase_err)]
    if finite_phase.size:
        bins_phase = 36
        counts, edges = np.histogram(finite_phase, bins=bins_phase, range=(-pi, pi), density=False)
        centers = (edges[:-1] + edges[1:]) / 2.0
        centers = (centers + 2*pi) % (2*pi)
        width = (2*pi) / bins_phase
        ax_polar.bar(centers, counts, width=width, bottom=0.0, align='center', color='tab:purple', alpha=0.75)

        plv = summary.get('plv')
        mean_phase = summary.get('phase_err_mean')
        if mean_phase is not None and np.isfinite(mean_phase):
            mean_phase = (mean_phase + 2*pi) % (2*pi)
            radius = max(counts.max(), 1.0) if counts.size else 1.0
            ax_polar.annotate(
                '',
                xy=(mean_phase, radius),
                xytext=(mean_phase, 0.0),
                arrowprops=dict(arrowstyle='->', color='tab:orange', linewidth=2.0)
            )
        text_lines = []
        if plv is not None and np.isfinite(plv):
            text_lines.append(f'PLV: {plv:.3f}')
        rayleigh_p = summary.get('rayleigh_p')
        if rayleigh_p is not None and np.isfinite(rayleigh_p):
            text_lines.append(f'Rayleigh p: {rayleigh_p:.3f}')
        phase_sd = summary.get('phase_err_sd')
        if phase_sd is not None and np.isfinite(phase_sd):
            text_lines.append(f'Phase SD: {phase_sd:.3f} rad')
        if text_lines:
            ax_polar.text(
                1,
                0.95,
                '\n'.join(text_lines),
                transform=ax_polar.transAxes,
                ha='left',
                va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, linewidth=0.0)
            )
    else:
        ax_polar.text(0.5, 0.5, 'No finite phase data', transform=ax_polar.transAxes, ha='center', va='center')

    ax_polar.set_title('Phase error distribution')
    ax_polar.set_theta_zero_location('N')
    ax_polar.set_theta_direction(-1)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

# Self-test
def _self_test():
    fs = 256.0
    T = 60.0
    t = np.arange(int(T*fs)) / fs
    f0 = 0.8
    so = 80*np.sin(2*np.pi*f0*t) + 10*np.sin(2*np.pi*0.1*t)
    noise = 5*np.random.randn(len(t))
    x = so + noise
    period = 1.25
    t_neg = np.arange(1.0, T-1.0, period)
    delay_true = period/2.0
    t_pos = t_neg + delay_true
    events = pd.DataFrame({'t_neg_s': t_neg, 't_pos_s': t_pos})
    delay_s = delay_true - 0.025
    summary, series = assess_targeting_performance(events, x, fs, delay_s, return_series=True)
    assert summary['n'] == len(events)
    assert np.isfinite(summary['sec_err_mean'])
    assert np.isfinite(summary['plv'])
    plot_targeting_diagnostics(
        series['sec_err'],
        series['phase_err'],
        summary,
        title="Synthetic targeting diagnostics",
    )
    return summary



def build_delay_series_time(events: pd.DataFrame,
                            win_len_s: float = 900.0,
                            step_s: float = 300.0,
                            robust: bool = True) -> pd.DataFrame:
    """Compute neg->pos delay in sliding time windows. Returns ['t_center_s','delay_s','n_events']."""
    if len(events) == 0:
        return pd.DataFrame(columns=['t_center_s','delay_s','n_events'])
    t = events['t_neg_s'].to_numpy(float)
    lat = (events['t_pos_s'] - events['t_neg_s']).to_numpy(float)
    t_min, t_max = np.nanmin(t), np.nanmax(t)
    centers = np.arange(t_min + win_len_s/2, t_max - win_len_s/2 + 1e-9, step_s)
    rows = []
    for c in centers:
        m = (t >= c - win_len_s/2) & (t < c + win_len_s/2)
        if not np.any(m): continue
        l = lat[m]; l = l[np.isfinite(l) & (l > 0) & (l < 2.0)]
        if l.size == 0: continue
        d = np.median(l) if robust else np.mean(l)
        rows.append({'t_center_s': float(c), 'delay_s': float(d), 'n_events': int(l.size)})
    return pd.DataFrame(rows)


def build_delay_series_ema(
    events: pd.DataFrame,
    *,
    half_life_s: float | None = 1800.0,
    fixed_alpha: float | None = None,           # <- new
    min_alpha: float = 0.05,
    robust_init: bool = True,
    max_latency_s: float | None = 2.0,
    outlier_k: float = 5.0,                     # <- new: residual gate (~5 MAD)
    track_variance: bool = True                 # <- new
) -> pd.DataFrame:
    """
    Time-aware EMA of neg->pos latency with optional robust gating and EW variance.
    Returns diagnostics for QA.
    """

    if len(events) == 0:
        return pd.DataFrame(columns=[
            't_center_s','delay_s','n_events','alpha','dt_s','latency_s',
            'residual','accepted','ew_sigma_lat','delay_sd'
        ])

    ev = events.sort_values('t_neg_s').reset_index(drop=True)
    t_neg = ev['t_neg_s'].to_numpy(dtype=float)
    lat = (ev['t_pos_s'] - ev['t_neg_s']).to_numpy(dtype=float)

    finite = np.isfinite(t_neg) & np.isfinite(lat) & (lat > 0.0)
    if max_latency_s is not None:
        finite &= (lat < float(max_latency_s))
    t_neg, lat = t_neg[finite], lat[finite]

    if t_neg.size == 0:
        return pd.DataFrame(columns=[
            't_center_s','delay_s','n_events','alpha','dt_s','latency_s',
            'residual','accepted','ew_sigma_lat','delay_sd'
        ])

    rows = []
    delay = None
    n_used = 0
    last_t = None

    # For time-based alpha
    log2 = np.log(2.0) if (half_life_s is not None and half_life_s > 0) else None
    use_fixed_alpha = (half_life_s is None and fixed_alpha is not None)

    # Robust-ish init stash
    init_vals: list[float] = []

    # EW moments for measurement noise (latency) and MAD proxy
    m = None   # EW mean of latency
    s = None   # EW second moment of latency
    beta = 0.1 # EW update for noise tracking (can expose)
    mad = None # EW MAD proxy

    for t_i, lat_i in zip(t_neg, lat):
        lat_i = float(lat_i)
        dt = float(t_i - last_t) if last_t is not None else 0.0
        if dt < 0:  # guard clock weirdness
            dt = 0.0

        # Choose alpha
        if use_fixed_alpha:
            alpha = float(np.clip(fixed_alpha, min_alpha, 1.0))
        elif log2 is not None and dt > 0:
            alpha = 1.0 - np.exp(-log2 * dt / max(half_life_s, 1e-9))
            alpha = float(np.clip(alpha, min_alpha, 1.0))
        else:
            alpha = float(min_alpha)

        accepted = True
        residual = None
        ew_sigma_lat = None
        delay_sd = None

        # Init phase
        if delay is None:
            init_vals.append(lat_i)
            if len(init_vals) >= 3:
                delay = float(np.median(init_vals) if robust_init else np.mean(init_vals))
                # bootstrap EW stats
                m = delay
                s = delay**2
                mad = np.median(np.abs(np.array(init_vals) - delay)) + 1e-9
            else:
                delay = lat_i
                if m is None:
                    m, s, mad = lat_i, lat_i**2, 1e-3
        else:
            # Robust residual gate using EW MAD
            residual = lat_i - delay
            if outlier_k is not None and mad is not None:
                if np.abs(residual) > outlier_k * mad:
                    accepted = False

            if accepted:
                delay = delay + alpha * (lat_i - delay)

        # Update EW noise trackers (always learn noise, even if not accepted)
        if track_variance and m is not None:
            m = m + beta * (lat_i - m)
            s = s + beta * (lat_i*lat_i - s)
            var_lat = max(s - m*m, 1e-9)
            ew_sigma_lat = float(np.sqrt(var_lat))
            # MAD proxy EW update (L1)
            mad = (1 - beta) * mad + beta * (np.abs(lat_i - m) + 1e-9)

            # Approximate sd of the EMA estimate at this step
            # (steady-state EWMA variance; decent heuristic for CIs)
            delay_sd = float(np.sqrt((alpha / max(2 - alpha, 1e-6)) * var_lat))

        n_used += int(accepted)
        last_t = t_i

        rows.append({
            't_center_s': float(t_i),
            'delay_s': float(delay),
            'n_events': int(n_used),
            'alpha': float(alpha),
            'dt_s': float(dt),
            'latency_s': float(lat_i),
            'residual': None if residual is None else float(residual),
            'accepted': bool(accepted),
            'ew_sigma_lat': ew_sigma_lat,
            'delay_sd': delay_sd
        })

    return pd.DataFrame(rows)


import numpy as np
import pandas as pd

def build_delay_series_kalman(
    events: pd.DataFrame,
    *,
    half_life_s: float = 1800.0,
    min_alpha: float = 0.05,
    beta: float = 0.2,    # EW update for measurement noise tracking
    robust_init: bool = True,
    max_latency_s: float = 1.0
) -> pd.DataFrame:
    """
    Kalman-filter version of time-varying smoothing for neg->pos latency.
    Same arguments and return columns as the EMA version.

    Model (scalar random walk with time-varying gain matched to EMA half-life):
        x_i      = true delay at event i
        z_i      = observed latency = t_pos - t_neg
        x_i      = x_{i-1} + w_i,    w_i ~ N(0, Q_i)
        z_i      = x_i + v_i,        v_i ~ N(0, R_i)

    We set a *target* EMA-like responsiveness K_target from (dt, half_life),
    then back out Q_i so that the Kalman gain equals (or closely matches)
    K_target given the current R_i and covariance P_{i-1}.

    Returns a DataFrame with columns:
        ['t_center_s', 'delay_s', 'n_events']
    """
    # Empty guard: keep exact column names/behavior
    if len(events) == 0:
        return pd.DataFrame(columns=['t_center_s', 'delay_s', 'n_events'])

    # Sort and compute raw latencies
    ev = events.sort_values('t_neg_s').reset_index(drop=True)
    t_neg = ev['t_neg_s'].to_numpy(dtype=float)
    lat   = (ev['t_pos_s'] - ev['t_neg_s']).to_numpy(dtype=float)

    # Finite, positive, and bounded latencies (mirror original)
    finite = np.isfinite(t_neg) & np.isfinite(lat) & (lat > 0.0)
    if max_latency_s is not None:
        finite &= (lat < float(max_latency_s))
    t_neg, lat = t_neg[finite], lat[finite]

    if t_neg.size == 0:
        return pd.DataFrame(columns=['t_center_s', 'delay_s', 'n_events'])

    # Half-life helper
    log2 = np.log(2.0) if (half_life_s and half_life_s > 0.0) else None

    rows = []
    n_used = 0
    last_t = None

    # Kalman state and variance
    x = None    # state (delay estimate)
    P = None    # state variance (posterior)

    # Online estimate of measurement variance R via EW moments
    # Keeps the interface clean (no extra args).
    m = None    # EW mean of latency
    s2 = None   # EW second moment (for variance)
    eps = 1e-9

    # Bootstrap cache for robust init
    init_vals = []

    for t_i, z_i in zip(t_neg, lat):
        z_i = float(z_i)
        # time since last event (for alpha / target gain)
        dt = float(t_i - last_t) if last_t is not None else 0.0
        if dt < 0:
            dt = 0.0

        # Target responsiveness (EMA-equivalent gain)
        if log2 is not None and dt > 0:
            K_target = 1.0 - np.exp(-log2 * dt / max(half_life_s, 1e-9))
            K_target = float(np.clip(K_target, min_alpha, 0.999))  # avoid 1.0 exactly
        else:
            K_target = float(np.clip(min_alpha, 1e-6, 0.999))

        if x is None:
            # --- Initialization phase: collect up to 3 values, robust option
            init_vals.append(z_i)
            if len(init_vals) >= 3:
                x0 = float(np.median(init_vals) if robust_init else np.mean(init_vals))
                x = x0
                # Initialize variance from bootstrap spread (MAD->variance)
                mad = np.median(np.abs(np.array(init_vals) - x0)) + eps
                P = float((1.4826 * mad)**2) if mad > 0 else 1e-3

                # Initialize measurement noise tracker near observed spread
                m = x0
                s2 = x0**2
            else:
                x = z_i
                P = 1e-3
                m = z_i
                s2 = z_i**2
        else:
            # --- Update EW measurement noise stats (on raw observations)
            #     R is derived from these stats and used below.
            m = m + beta * (z_i - m)
            s2 = s2 + beta * (z_i*z_i - s2)
            R = max(s2 - m*m, 1e-9)  # estimated measurement variance

            # --- Choose Q to achieve (approximately) K_target this step.
            # Kalman gain for scalar RW model is:
            #   K = P_pred / (P_pred + R),  where P_pred = P + Q
            # Solve for the P_pred that yields K_target:
            #   P_pred = (K_target / (1 - K_target)) * R
            P_pred_target = (K_target / max(1.0 - K_target, 1e-6)) * R

            # Then Q = P_pred_target - P (clip to nonnegative)
            Q = max(P_pred_target - P, 0.0)

            # --- Predict
            P_pred = P + Q  # x_pred = x (random walk), so no drift on the mean

            # --- Update
            K = P_pred / (P_pred + R)
            # (K should be close to K_target; tiny differences due to clipping)
            x = x + K * (z_i - x)
            P = (1.0 - K) * P_pred

        n_used += 1
        last_t = t_i
        rows.append({'t_center_s': float(t_i), 'delay_s': float(x), 'n_events': int(n_used)})

    return pd.DataFrame(rows)


def assess_targeting_performance_adaptive(events: pd.DataFrame,
                                          signal: np.ndarray,
                                          fs: float,
                                          delay_series: pd.DataFrame,
                                          phase_band=(0.16, 1.25),
                                          return_series=False):
    """Assess performance using a time-varying delay series (nearest-neighbor mapping)."""
    if len(delay_series) == 0 or len(events) == 0:
        return {'n': 0}, None
    t_neg = events['t_neg_s'].to_numpy(float)
    ds_t = delay_series['t_center_s'].to_numpy(float)
    ds_d = delay_series['delay_s'].to_numpy(float)
    idx = np.abs(t_neg[:, None] - ds_t[None, :]).argmin(axis=1)
    delays = ds_d[idx]
    t_pos = events['t_pos_s'].to_numpy(float)
    t_click = t_neg + delays
    sec_err = t_click - t_pos

    # phase via Hilbert (robust version you have)
    phi = _band_hilbert_phase_safe(signal, fs, f_lo=phase_band[0], f_hi=phase_band[1], order=4)
    phi_click = _interp_at_times(phi, fs, t_click) if np.any(np.isfinite(phi)) else np.full_like(sec_err, np.nan)
    phi_pos   = _interp_at_times(phi, fs, t_pos)   if np.any(np.isfinite(phi)) else np.full_like(sec_err, np.nan)
    phase_err = np.angle(np.exp(1j*(phi_click - phi_pos)))
    if not np.any(np.isfinite(phase_err)) and all(c in events.columns for c in ['t0_s','t1_s']):
        T = (events['t1_s'] - events['t0_s']).to_numpy(float)
        T = np.where(T <= 0, np.nan, T)
        phase_err = (2*np.pi * (sec_err / T) + np.pi) % (2*np.pi) - np.pi

    def _iqr(x): 
        q75, q25 = np.percentile(x, [75, 25]) if x.size else (np.nan, np.nan)
        return q75 - q25
    def _hit_rate(x, tol):
        return float(np.mean(np.abs(x) <= tol)) if x.size else np.nan

    n = sec_err.size
    summary = {
        'n': int(n),
        'sec_err_mean': float(np.mean(np.abs(sec_err))) if n else np.nan,
        'sec_err_median': float(np.median(np.abs(sec_err))) if n else np.nan,
        'sec_err_iqr': float(_iqr(np.abs(sec_err))) if n else np.nan,
        'sec_hit_rate_25ms': _hit_rate(np.abs(sec_err), 0.025),
        'sec_hit_rate_50ms': _hit_rate(np.abs(sec_err), 0.050),
        'sec_hit_rate_75ms': _hit_rate(np.abs(sec_err), 0.075),
        'sec_hit_rate_100ms': _hit_rate(np.abs(sec_err), 0.100),
        'sec_hit_rate_250ms': _hit_rate(np.abs(sec_err), 0.250),
    }
    finite = np.isfinite(phase_err)
    if np.any(finite):
        pe = phase_err[finite]
        plv = _plv(pe); Z, p = _rayleigh_test(pe)
        circ_mean = _circular_mean(pe)
        circ_sd = float(np.sqrt(2*(1 - plv))) if np.isfinite(plv) else np.nan
        summary.update({'plv': float(plv) if np.isfinite(plv) else np.nan,
                        'rayleigh_Z': float(Z) if np.isfinite(Z) else np.nan,
                        'rayleigh_p': p,
                        'phase_err_mean': float(circ_mean) if np.isfinite(circ_mean) else np.nan,
                        'phase_err_sd': circ_sd})
    else:
        summary.update({'plv': np.nan, 'rayleigh_Z': np.nan, 'rayleigh_p': np.nan,
                        'phase_err_mean': np.nan, 'phase_err_sd': np.nan})

    return (summary, {'sec_err': sec_err, 'phase_err': phase_err, 't_click': t_click}) if return_series else (summary, None)



def compare_fixed_vs_adaptive(events,
                              signal,
                              fs,
                              fixed_delay_s,
                              phase_band=(0.16, 1.25),
                              delay_series_fn=None,
                              delay_series_kwargs=None):
    fixed_perf, _ = assess_targeting_performance(events, signal, fs, fixed_delay_s,
                                                 phase_band=phase_band, return_series=False)

    if delay_series_fn is None:
        delay_series_fn = build_delay_series_ema
    kwargs = delay_series_kwargs or {}
    dser = delay_series_fn(events, **kwargs)
    adapt_perf, _ = assess_targeting_performance_adaptive(events, signal, fs, dser,
                                                          phase_band=phase_band, return_series=False)
    return {'fixed': fixed_perf, 'adaptive': adapt_perf, 'delay_series': dser}


def main():
    _selftest_summary = _self_test()
    print(_selftest_summary)
    
if __name__ == "__main__":
    main()
