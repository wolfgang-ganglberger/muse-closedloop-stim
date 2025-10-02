"""Run the slow-wave detection pipeline against a sample Muse recording."""

from __future__ import annotations

from pathlib import Path

from run_sw_detection_pipeline import run_pipeline

DATA_DIR = Path(
    "/Users/wolfgang/cdac Dropbox/a_People_BIDMC/WolfgangGanglberger/Muse/data_david/slow_waves_2025_09"
)
FS = 256.0

SAMPLE_DATA_DIR: Path = DATA_DIR  # Change to your sample data directory if needed.
SAMPLE_FILE_NAME: str | None = None  # Set to a specific filename to override the default selection.


def main() -> None:
    run_pipeline(
        data_dir=SAMPLE_DATA_DIR,
        file_name=SAMPLE_FILE_NAME,
        fs=FS,
        plot_full_overview=True,
    )

if __name__ == "__main__":
    main()
