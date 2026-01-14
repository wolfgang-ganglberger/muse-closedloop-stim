from __future__ import annotations
import time, threading, queue, math, sys
from typing import Sequence, Optional

import numpy as np
from scipy.signal import butter, lfilter
from pylsl import StreamInfo, StreamOutlet

from osc_receiver import MuseOSCReceiver


# ----------------------------------------------------------------------
# Detector thread -----------------------------------------------------
class _Kalman1D:
    """Minimal 1D Kalman filter for online delay estimation (seconds)."""
    def __init__(self, x0: float, p0: float = 0.1, q: float = 1e-4, r: float = 1e-2):
        self.x = float(x0)
        self.p = float(p0)
        self.q = float(q)
        self.r = float(r)

    def update(self, z: float) -> float:
        # Predict
        self.p += self.q
        # Update
        k = self.p / (self.p + self.r)
        self.x = self.x + k * (z - self.x)
        self.p = (1.0 - k) * self.p
        return self.x


class _SlowWaveDetector:
    """
    Online slow-wave detector (v1).
    Detects negative peaks in a 0.5-4 Hz signal and schedules stimulation
    near the positive peak using a Kalman-estimated neg->pos delay.
    """
    def __init__(
        self,
        *,
        fs: int,
        refractory_s: float = 6.0,          # local refractory after a fire
        dur_min_s: float = 0.8,             # min cycle duration (pos->neg to pos->neg)
        dur_max_s: float = 2.0,             # max cycle duration (pos->neg to pos->neg)
        neg_mult: float = 1.05,             # neg-peak threshold multiplier vs EWMA mean
        min_stats_events: int = 20,         # events required before using EWMA stats
        ewma_alpha: float = 0.02,           # EWMA update rate for neg peak stats
        fallback_neg_uV: float = -30.0,     # used before EWMA stats are stable
        fallback_pp_uV: float = 60.0,      # reserved for future pp gating
        min_neg_interval_s: float = 0.8,    # minimum time between neg peaks (debounce)
        upstate_window: Sequence[float] = (-0.15, 0.35),  # window around pos-peak target
        peak_advance_s: float = 0.01,       # fire slightly before expected pos peak (parameter in seconds)
        device_latency_s: float = 0.2,      # hardware latency to subtract
        global_on_s: float = 12.0,           # global ON duration (seconds)
        global_off_s: float = 10.0,          # global OFF duration (seconds)
        kalman_p0: float = 0.1,             # Kalman initial variance
        kalman_q: float = 1e-4,             # Kalman process noise
        kalman_r: float = 1e-2,             # Kalman measurement noise
        delay_init_s: float = 0.5,          # initial neg->pos delay estimate
        delay_min_s: float = 0.1,           # clamp min neg->pos delay
        delay_max_s: float = 1.5,           # clamp max neg->pos delay
        use_stage_gate: bool = False,      # require N2/N3 stage gate for global window
        stage_prob_min_n2: float = 0.75,    # allow if P(N2) exceeds this
        stage_prob_sum_n2_n3: float = 0.90, # or if P(N2)+P(N3) exceeds this
        stage_prob_n2_min_for_sum: float = 0.30,  # and P(N2) exceeds this
        use_signal_quality_gate: bool = False,    # require SQC gate for global window
        signal_quality_min: float = 0.60,         # SQC threshold (0..1)
    ):
        self.fs = float(fs)
        self.refractory_s = float(refractory_s)
        self.dur_min_s = float(dur_min_s)
        self.dur_max_s = float(dur_max_s)
        self.neg_mult = float(neg_mult)
        self.min_stats_events = int(min_stats_events)
        self.ewma_alpha = float(ewma_alpha)
        self.fallback_neg_uV = float(fallback_neg_uV)
        self.fallback_pp_uV = float(fallback_pp_uV)
        self.min_neg_interval_s = float(min_neg_interval_s)
        self.upstate_window = (float(upstate_window[0]), float(upstate_window[1]))
        self.peak_advance_s = float(peak_advance_s)
        self.device_latency_s = float(device_latency_s)
        self.global_on_s = float(global_on_s)
        self.global_off_s = float(global_off_s)
        self.delay_min_s = float(delay_min_s)
        self.delay_max_s = float(delay_max_s)
        self.use_stage_gate = bool(use_stage_gate)
        self.stage_prob_min_n2 = float(stage_prob_min_n2)
        self.stage_prob_sum_n2_n3 = float(stage_prob_sum_n2_n3)
        self.stage_prob_n2_min_for_sum = float(stage_prob_n2_min_for_sum)
        self.use_signal_quality_gate = bool(use_signal_quality_gate)
        self.signal_quality_min = float(signal_quality_min)

        self._delay_kf = _Kalman1D(delay_init_s, p0=kalman_p0, q=kalman_q, r=kalman_r)

        self._last_fire = -math.inf
        self._last_neg_time = -math.inf
        self._pending_window: Optional[tuple[float, float]] = None
        self._global_t0 = None

        self._mean_neg = None
        self._mean_pp = None
        self._n_stats = 0

        self._in_cycle = False
        self._cycle_t0 = None
        self._cycle_min = math.inf
        self._cycle_max = -math.inf
        self._neg_detected = False
        self._pos_detected = False
        self._neg_peak_time = None
        self._pos_peak_time = None

        self._prev = None
        self._prev_t = None
        self._prev2 = None
        self._prev2_t = None
        self._stage_n2 = None
        self._stage_n3 = None
        self._sqc_ch0 = None
        self._sqc_ch3 = None

    def update_stage_probs(self, n2: float | None, n3: float | None) -> None:
        if n2 is None or n3 is None:
            return
        self._stage_n2 = float(n2)
        self._stage_n3 = float(n3)

    def update_signal_quality(self, ch0: float | None, ch3: float | None) -> None:
        if ch0 is None or ch3 is None:
            return
        self._sqc_ch0 = float(ch0)
        self._sqc_ch3 = float(ch3)

    def _time_gate_open(self, t_now: float) -> bool:
        if self.global_on_s <= 0:
            return True
        cycle_len = self.global_on_s + max(0.0, self.global_off_s)
        if cycle_len <= 0:
            return True
        if self._global_t0 is None:
            self._global_t0 = float(t_now)
        phase = (float(t_now) - self._global_t0) % cycle_len
        return phase < self.global_on_s

    def _stage_gate_open(self) -> bool:
        if not self.use_stage_gate:
            return True
        if self._stage_n2 is None or self._stage_n3 is None:
            return False
        n2 = float(self._stage_n2)
        n3 = float(self._stage_n3)
        if not np.isfinite(n2) or not np.isfinite(n3):
            return False
        if n2 > self.stage_prob_min_n2:
            return True
        if (n2 + n3) > self.stage_prob_sum_n2_n3 and n2 > self.stage_prob_n2_min_for_sum:
            return True
        return False

    def _signal_quality_gate_open(self) -> bool:
        if not self.use_signal_quality_gate:
            return True
        if self._sqc_ch0 is None or self._sqc_ch3 is None:
            return False
        ch0 = float(self._sqc_ch0)
        ch3 = float(self._sqc_ch3)
        if not np.isfinite(ch0) or not np.isfinite(ch3):
            return False
        return ch0 > self.signal_quality_min and ch3 > self.signal_quality_min

    def _global_gate_open(self, t_now: float) -> bool:
        return self._time_gate_open(t_now) and self._stage_gate_open() and self._signal_quality_gate_open()

    def _current_neg_thresh(self) -> float:
        if self._n_stats >= self.min_stats_events and self._mean_neg is not None and self._mean_neg < 0:
            return self.neg_mult * self._mean_neg
        return self.fallback_neg_uV

    def _update_stats(self, neg_peak: float, pp: float) -> None:
        if not np.isfinite(neg_peak) or not np.isfinite(pp) or pp <= 0:
            return
        if self._mean_neg is None:
            self._mean_neg = float(neg_peak)
            self._mean_pp = float(pp)
        else:
            a = self.ewma_alpha
            self._mean_neg = (1.0 - a) * self._mean_neg + a * float(neg_peak)
            self._mean_pp = (1.0 - a) * self._mean_pp + a * float(pp)
        self._n_stats += 1

    def _schedule_window(self, t_neg: float) -> None:
        delay_s = float(self._delay_kf.x)
        delay_s = min(self.delay_max_s, max(self.delay_min_s, delay_s))
        win_lo, win_hi = self.upstate_window
        start = t_neg + delay_s + win_lo - self.device_latency_s - self.peak_advance_s
        end = t_neg + delay_s + win_hi - self.device_latency_s - self.peak_advance_s
        if end <= t_neg:
            return
        self._pending_window = (start, end)

    def _maybe_fire(self, t_now: float) -> Optional[float]:
        if self._pending_window is None:
            return None
        start, end = self._pending_window
        if t_now > end:
            self._pending_window = None
            return None
        if t_now < start:
            return None
        if (t_now - self._last_fire) < self.refractory_s:
            return None
        self._last_fire = t_now
        self._pending_window = None
        return t_now

    def _start_cycle(self, t0: float, y0: float) -> None:
        self._in_cycle = True
        self._cycle_t0 = t0
        self._cycle_min = float(y0)
        self._cycle_max = float(y0)
        self._neg_detected = False
        self._pos_detected = False
        self._neg_peak_time = None
        self._pos_peak_time = None

    def _finalize_cycle(self, t1: float) -> None:
        if not self._in_cycle or self._cycle_t0 is None:
            return
        dur = t1 - float(self._cycle_t0)
        if self.dur_min_s <= dur <= self.dur_max_s:
            neg_peak = float(self._cycle_min)
            pp = float(self._cycle_max - self._cycle_min)
            self._update_stats(neg_peak, pp)
        self._in_cycle = False

    def update(self, t: np.ndarray, y: np.ndarray) -> list[float]:
        fires: list[float] = []
        states: list[list[float]] = []
        for t_now, y_now in zip(t, y):
            t_now = float(t_now)
            window_active = 0
            if self._pending_window is not None:
                start, end = self._pending_window
                if start <= t_now <= end:
                    window_active = 1
            stage_on = 1 if self._stage_gate_open() else 0
            sqc_on = 1 if self._signal_quality_gate_open() else 0
            global_on = 1 if self._global_gate_open(t_now) else 0
            states.append([t_now, float(global_on), float(stage_on), float(sqc_on), float(window_active)])

            fired = self._maybe_fire(t_now)
            if fired is not None:
                fires.append(fired)

            if self._prev is not None:
                prev = float(self._prev)
                if prev >= 0.0 and float(y_now) < 0.0:
                    dt = float(t_now) - float(self._prev_t)
                    if dt <= 0:
                        t_cross = float(t_now)
                    else:
                        frac = prev / (prev - float(y_now))
                        t_cross = float(self._prev_t) + frac * dt
                    if self._in_cycle:
                        self._finalize_cycle(t_cross)
                    self._start_cycle(t_cross, float(y_now))

            if self._prev2 is not None and self._in_cycle:
                prev2 = float(self._prev2)
                prev = float(self._prev)
                curr = float(y_now)
                if (not self._neg_detected) and (prev < prev2) and (prev < curr) and (prev < 0.0):
                    t_neg = float(self._prev_t)
                    neg_thresh = self._current_neg_thresh()
                    if (t_neg - self._last_neg_time) >= self.min_neg_interval_s and prev <= neg_thresh:
                        self._neg_detected = True
                        self._neg_peak_time = t_neg
                        self._last_neg_time = t_neg
                        if self._global_gate_open(t_neg) and self._pending_window is None:
                            self._schedule_window(t_neg)
                if self._neg_detected and (not self._pos_detected) and (prev > prev2) and (prev > curr) and (prev > 0.0):
                    t_pos = float(self._prev_t)
                    if self._neg_peak_time is not None:
                        delay = t_pos - float(self._neg_peak_time)
                        if self.delay_min_s <= delay <= self.delay_max_s:
                            self._delay_kf.update(delay)
                            self._pos_detected = True
                            self._pos_peak_time = t_pos

            if self._in_cycle:
                self._cycle_min = min(self._cycle_min, float(y_now))
                self._cycle_max = max(self._cycle_max, float(y_now))

            self._prev2, self._prev2_t = self._prev, self._prev_t
            self._prev, self._prev_t = float(y_now), float(t_now)
        return fires, np.asarray(states, dtype=np.float32)


class DetectorThread(threading.Thread):
    """
    Pulls latest EEG from MuseOSCReceiver, runs causal band-pass,
    fires stimulus based on selected mode:
      - "simple-threshold": abs(signal) crosses threshold
      - "slow-wave-v1": online slow-oscillation detection
    """
    def __init__(self, rx: MuseOSCReceiver, out_q: "queue.Queue[float]",
                 fs: int = 256, band: Sequence[float] = (0.5, 4),
                 thresh_uV: float = 100.0, refractory_s: float = 1.5,
                 use_virtual_frontal: bool = True,
                 virtual_ch_idx: Sequence[int] = (0, 3),
                 mode: str = "simple-threshold",
                 slow_wave_params: Optional[dict] = None,
                 state_lsl_name: Optional[str] = None,
                 state_channel_names: Sequence[str] = ("global_gate_on", "stage_gate_on", "signal_quality_on", "stim_window_on")):
        super().__init__(daemon=True)
        self.rx, self.out_q = rx, out_q
        self.fs, self.thresh = fs, thresh_uV
        self.refractory_s = refractory_s
        self._last_fire = -math.inf
        self._last_t_processed = None
        self.use_virtual_frontal = use_virtual_frontal
        self.virtual_ch_idx = tuple(virtual_ch_idx)
        self.mode = mode

        # causal Butterworth
        b, a = butter(2, np.array(band) / (0.5 * fs), btype='bandpass')
        self._zi = np.zeros(max(len(a), len(b)) - 1)   # filter state
        self._ba = (b, a)
        self._sw = None
        self._state_out = None
        if self.mode == "slow-wave-v1":
            params = dict(slow_wave_params or {})
            params.setdefault("fs", fs)
            params.setdefault("refractory_s", max(2.0, refractory_s))
            self._sw = _SlowWaveDetector(**params)
            if state_lsl_name:
                info = StreamInfo(
                    state_lsl_name,
                    "StimState",
                    len(state_channel_names),
                    fs,
                    "float32",
                    "stim_state_stream",
                )
                chns = info.desc().append_child("channels")
                for lab in state_channel_names:
                    ch = chns.append_child("channel")
                    ch.append_child_value("label", str(lab))
                self._state_out = StreamOutlet(info, chunk_size=64, max_buffered=fs * 60)

    def run(self):
        if self.use_virtual_frontal:
            ch_slice = list(self.virtual_ch_idx)   # TP9/TP10 indices in /eeg
        else:
            ch_slice = slice(0, 1)   # use first EEG channel
        while True:
            eeg = self.rx.get('/eeg', ch_slice)
            if eeg.shape[0] < 2:
                time.sleep(0.05); continue
            if self._last_t_processed is None:
                new = eeg
            else:
                new = eeg[eeg[:, 0] > self._last_t_processed]
            if new.shape[0] == 0:
                time.sleep(0.01)
                continue
            t = new[-1, 0]

            if self.use_virtual_frontal:
                if new.shape[1] < 3:
                    time.sleep(0.05); continue
                tp9 = new[:, 1]
                tp10 = new[:, 2]
                sig = 0.5 * (tp9 + tp10)  # causal virtual frontal approximation
            else:
                sig = new[:, 1]

            # filter latest chunk
            y, self._zi = lfilter(*self._ba, sig, zi=self._zi)

            if self._sw is not None:
                metrics = self.rx.get("/muse_metrics")
                if metrics.size:
                    row = metrics[-1]
                    if row.shape[0] >= 20:
                        sqc_ch0 = float(row[1 + 0])
                        sqc_ch3 = float(row[1 + 3])
                        n2 = float(row[1 + 16])
                        n3 = float(row[1 + 17])
                        if max(n2, n3) > 1.5:
                            n2 /= 100.0
                            n3 /= 100.0
                        if max(sqc_ch0, sqc_ch3) > 1.5:
                            sqc_ch0 /= 100.0
                            sqc_ch3 /= 100.0
                        self._sw.update_stage_probs(n2, n3)
                        self._sw.update_signal_quality(sqc_ch0, sqc_ch3)

            if self.mode == "simple-threshold":
                if (t - self._last_fire) >= self.refractory_s:
                    detection_window_samples = min(5, len(y))
                    if detection_window_samples > 0 and np.max(np.abs(y[-detection_window_samples:])) > self.thresh:
                        self._last_fire = t
                        self.out_q.put(t)
            elif self.mode == "slow-wave-v1" and self._sw is not None:
                fires, states = self._sw.update(new[:, 0], y)
                for t_fire in fires:
                    self.out_q.put(t_fire)
                if self._state_out is not None and states.size:
                    samples = states[:, 1:].tolist()
                    ts_list = states[:, 0].tolist()
                    self._state_out.push_chunk(samples, ts_list)
            else:
                raise ValueError(f"Unknown detection mode: {self.mode}")

            self._last_t_processed = float(t)
            time.sleep(0.01)
