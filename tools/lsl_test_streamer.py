"""Simple LSL test streamer for local development."""

from __future__ import annotations

import argparse
import math
import time
from typing import Iterable

import numpy as np
from pylsl import StreamInfo, StreamOutlet


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit a synthetic LSL stream for testing.")
    parser.add_argument("--name", default="TestEEG", help="LSL stream name")
    parser.add_argument("--type", default="EEG", help="LSL stream type")
    parser.add_argument("--channels", type=int, default=4, help="Number of channels")
    parser.add_argument("--srate", type=float, default=256.0, help="Sampling rate (Hz)")
    parser.add_argument("--source-id", default="test_eeg_source", help="LSL source_id")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    info = StreamInfo(
        args.name,
        args.type,
        args.channels,
        args.srate,
        "float32",
        args.source_id,
    )
    outlet = StreamOutlet(info, chunk_size=32, max_buffered=60)

    phase = 0.0
    dt = 1.0 / args.srate
    print(f"Streaming '{args.name}' ({args.type}) @ {args.srate} Hz. Ctrl-C to stop.")
    try:
        while True:
            chunk = []
            for _ in range(32):
                row = [
                    math.sin(phase + 0.2 * idx) + 0.05 * np.random.randn()
                    for idx in range(args.channels)
                ]
                chunk.append(row)
                phase += 2 * math.pi * dt * 10.0
            outlet.push_chunk(chunk)
            time.sleep(32 * dt)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
