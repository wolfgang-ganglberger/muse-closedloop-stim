"""LSL subscriber for live EEG monitoring and analysis."""

from __future__ import annotations

import argparse
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
from pylsl import StreamInlet, resolve_byprop

logger = logging.getLogger(__name__)


def parse_channel_list(spec: str | None) -> list[int] | None:
    if not spec:
        return None
    channels: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s.strip())
            end = int(end_s.strip())
            if end < start:
                raise ValueError(f"Invalid channel range '{part}'")
            channels.extend(range(start, end + 1))
        else:
            channels.append(int(part))
    return sorted(set(channels))


def compute_features(samples: np.ndarray) -> dict[str, np.ndarray]:
    if samples.size == 0:
        return {}
    return {
        "mean": np.mean(samples, axis=0),
        "std": np.std(samples, axis=0),
        "rms": np.sqrt(np.mean(samples ** 2, axis=0)),
        "ptp": np.ptp(samples, axis=0),
    }


def _extract_channel_labels(info) -> list[str]:
    labels: list[str] = []
    desc = info.desc()
    ch = desc.child("channels").child("channel")
    while ch.valid():
        label = ch.child_value("label")
        labels.append(label if label else f"CH{len(labels):02d}")
        ch = ch.next_sibling()
    return labels


@dataclass
class BufferSnapshot:
    timestamps: np.ndarray
    samples: np.ndarray


class LSLSubscriber(threading.Thread):
    """Background LSL subscriber with a bounded in-memory buffer."""

    def __init__(
        self,
        *,
        lsl_type: str = "EEG",
        lsl_name: str | None = None,
        lsl_source_id: str | None = None,
        channels: list[int] | None = None,
        buffer_seconds: float = 10.0,
        resolve_interval_s: float = 2.0,
    ) -> None:
        super().__init__(daemon=True)
        self.lsl_type = lsl_type
        self.lsl_name = lsl_name
        self.lsl_source_id = lsl_source_id
        self.channels = channels
        self.buffer_seconds = float(buffer_seconds)
        self.resolve_interval_s = float(resolve_interval_s)

        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._buffer: deque[tuple[np.ndarray, np.ndarray]] = deque()
        self._total_samples = 0
        self._max_samples: int | None = None
        self._inlet: StreamInlet | None = None
        self._info = None
        self._channel_labels: list[str] = []

    def stop(self) -> None:
        self._stop.set()

    @property
    def channel_labels(self) -> list[str]:
        return self._channel_labels

    @property
    def info(self):
        return self._info

    def get_buffer(self) -> BufferSnapshot:
        with self._lock:
            if not self._buffer:
                return BufferSnapshot(np.array([]), np.empty((0, 0)))
            timestamps = np.concatenate([item[0] for item in self._buffer])
            samples = np.concatenate([item[1] for item in self._buffer], axis=0)
        return BufferSnapshot(timestamps, samples)

    def _resolve_stream(self):
        if self.lsl_source_id:
            results = resolve_byprop("source_id", self.lsl_source_id, timeout=2.0)
        else:
            results = resolve_byprop("type", self.lsl_type, timeout=2.0)
            if self.lsl_name:
                results = [info for info in results if info.name() == self.lsl_name]
        return results[0] if results else None

    def _configure_inlet(self, info) -> None:
        self._info = info
        channel_count = info.channel_count()
        labels = _extract_channel_labels(info)
        if not labels or len(labels) != channel_count:
            labels = [f"CH{idx:02d}" for idx in range(channel_count)]
        self._channel_labels = labels
        if self.channels:
            invalid = [idx for idx in self.channels if idx < 0 or idx >= channel_count]
            if invalid:
                raise ValueError(f"Invalid channel index/indices: {invalid}")
        nominal_srate = info.nominal_srate()
        if nominal_srate and nominal_srate > 0:
            self._max_samples = int(round(nominal_srate * self.buffer_seconds))
        else:
            self._max_samples = None
        self._inlet = StreamInlet(info, max_buflen=max(1, int(self.buffer_seconds * 2)))

    def _trim_by_samples(self) -> None:
        if self._max_samples is None:
            return
        while self._buffer and self._total_samples > self._max_samples:
            ts, data = self._buffer.popleft()
            self._total_samples -= len(ts)

    def _trim_by_time(self, latest_ts: float) -> None:
        if self._max_samples is not None:
            return
        threshold = latest_ts - self.buffer_seconds
        while self._buffer:
            ts, data = self._buffer[0]
            if ts[-1] >= threshold:
                break
            self._buffer.popleft()
            self._total_samples -= len(ts)

    def _append_chunk(self, timestamps: np.ndarray, samples: np.ndarray) -> None:
        with self._lock:
            self._buffer.append((timestamps, samples))
            self._total_samples += len(timestamps)
            self._trim_by_samples()
            self._trim_by_time(float(timestamps[-1]))

    def run(self) -> None:
        while not self._stop.is_set():
            if self._inlet is None:
                info = self._resolve_stream()
                if info is None:
                    time.sleep(self.resolve_interval_s)
                    continue
                try:
                    self._configure_inlet(info)
                    logger.info(
                        "Connected to LSL stream name='%s' type='%s' source_id='%s'",
                        info.name(),
                        info.type(),
                        info.source_id(),
                    )
                except Exception as exc:
                    logger.warning("Failed to configure inlet: %s", exc)
                    self._inlet = None
                    time.sleep(self.resolve_interval_s)
                    continue

            try:
                samples, timestamps = self._inlet.pull_chunk(timeout=0.5)
            except Exception as exc:
                logger.warning("LSL inlet error: %s", exc)
                self._inlet = None
                time.sleep(self.resolve_interval_s)
                continue

            if not timestamps:
                continue

            ts = np.asarray(timestamps, dtype=np.float64)
            data = np.asarray(samples, dtype=np.float32)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            if self.channels:
                data = data[:, self.channels]
            self._append_chunk(ts, data)


def _format_feature_vector(features: dict[str, np.ndarray]) -> str:
    if not features:
        return "no data"
    return " | ".join(
        f"{key}=" + ",".join(f"{val:.3f}" for val in values)
        for key, values in features.items()
    )


def _run_plot(subscriber: LSLSubscriber, channels: list[int] | None) -> None:
    import matplotlib.pyplot as plt

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 4))
    lines: list = []

    while subscriber.info is None:
        time.sleep(0.1)

    if channels:
        labels = [subscriber.channel_labels[idx] for idx in channels]
    else:
        labels = subscriber.channel_labels

    for label in labels:
        line, = ax.plot([], [], label=label)
        lines.append(line)

    ax.set_title("LSL EEG (live)")
    ax.set_xlabel("Seconds (relative)")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.show()

    try:
        while True:
            snap = subscriber.get_buffer()
            if snap.samples.size > 0:
                t0 = snap.timestamps[0]
                x = snap.timestamps - t0
                for idx, line in enumerate(lines):
                    if idx < snap.samples.shape[1]:
                        line.set_data(x, snap.samples[:, idx])
                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw_idle()
            plt.pause(0.05)
    except KeyboardInterrupt:
        pass


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Subscribe to an LSL stream for live monitoring.")
    parser.add_argument("--lsl-type", default="EEG", help="LSL stream type (default: EEG)")
    parser.add_argument("--lsl-name", default=None, help="Optional LSL stream name filter")
    parser.add_argument("--lsl-source-id", default=None, help="Optional LSL source_id filter")
    parser.add_argument("--channels", default=None, help="Channel list or range, e.g. '0-7,10,12'")
    parser.add_argument("--buffer-seconds", type=float, default=10.0, help="Ring buffer length in seconds")
    parser.add_argument("--mode", choices=("raw", "features", "both"), default="raw")
    parser.add_argument("--plot", action="store_true", help="Enable live Matplotlib plot")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s: %(message)s")
    channels = parse_channel_list(args.channels)

    subscriber = LSLSubscriber(
        lsl_type=args.lsl_type,
        lsl_name=args.lsl_name,
        lsl_source_id=args.lsl_source_id,
        channels=channels,
        buffer_seconds=args.buffer_seconds,
    )
    subscriber.start()

    if args.plot:
        _run_plot(subscriber, channels)
        subscriber.stop()
        return 0

    last_report = time.time()
    try:
        while True:
            time.sleep(0.1)
            if time.time() - last_report < 1.0:
                continue
            last_report = time.time()
            snap = subscriber.get_buffer()
            if snap.samples.size == 0:
                logger.info("Waiting for samples...")
                continue
            if args.mode in ("raw", "both"):
                logger.info(
                    "Last sample (LSL ts=%.3f): %s",
                    snap.timestamps[-1],
                    np.array2string(snap.samples[-1], precision=3),
                )
            if args.mode in ("features", "both"):
                features = compute_features(snap.samples)
                logger.info("Features: %s", _format_feature_vector(features))
    except KeyboardInterrupt:
        pass
    finally:
        subscriber.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
