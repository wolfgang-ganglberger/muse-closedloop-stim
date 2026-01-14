from __future__ import annotations

import time
import sys

import numpy as np
import sounddevice as sd

sys.path.append("./closedloop-utils")
from stimulus_player import build_stim


def main():
    fs_out = 48_000
    stim_rms_start = 0.02
    stim_rms_end = 1
    stim_rms_step = 0.02
    rms_vals = np.arange(stim_rms_start, stim_rms_end + 1e-9, stim_rms_step)

    stim_type = 'pink'  # 'pink' or 'morlet', depending on your experimental design
    stim_rms = 0.05  # RMS amplitude of the stimulus sound. This is overridden by the sweep.
    stim_peak = 0.20  # Peak cap or target (full-scale). Set to None to disable.
    stim_peak_normalize = False  # True = always normalize to stim_peak; False = only limit if exceeded.
    stim_dur = 0.050  # Duration of the stimulus sound in seconds

    print("Pink noise RMS sweep. Ctrl-C to stop.")
    try:
        for rms in rms_vals:
            stim_rms = float(rms)
            stim = build_stim(
                stim_type=stim_type,
                stim_dur=stim_dur,
                fs_out=fs_out,
                stim_rms=stim_rms,
                stim_peak=stim_peak,
                stim_peak_normalize=stim_peak_normalize,
            )
            print(f"Playing {stim_type}: stim_rms={stim_rms:.2f}, stim_peak={stim_peak}, peak_norm={stim_peak_normalize}")
            sd.play(stim, samplerate=fs_out, blocking=True)
            time.sleep(0.3)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        sd.stop()


if __name__ == "__main__":
    main()
