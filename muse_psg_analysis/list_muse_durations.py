"""Utility to list the duration of Muse EDF recordings."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Iterator, Tuple

import mne


DEFAULT_DATA_DIR = Path(
    "/Users/wolfgang/cdac Dropbox/a_People_BIDMC/WolfgangGanglberger/muse_data_scn_montreal/Simultaneous Muse-PSG/"
)


def iter_muse_edfs(base_dir: Path) -> Iterator[Tuple[str, Path]]:
    """Yield subject identifier and path for each Muse EDF recording."""
    for subject_dir in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        muse_dir = subject_dir / "muse"
        if not muse_dir.is_dir():
            continue
        for edf_path in sorted(muse_dir.glob("*_muse.edf")):
            yield subject_dir.name, edf_path


def duration_hours(edf_path: Path) -> float:
    """Return the recording duration in hours for a Muse EDF file."""
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
    try:
        total_seconds = raw.n_times / float(raw.info["sfreq"])
        return total_seconds / 3600.0
    finally:
        raw.close()


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Parent directory containing subject folders with Muse EDF files",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    base_dir = args.data_dir or DEFAULT_DATA_DIR
    if not base_dir.exists():
        print(f"Data directory not found: {base_dir}")
        return 1

    any_found = False
    for subject_id, edf_path in iter_muse_edfs(base_dir):
        any_found = True
        try:
            hours = duration_hours(edf_path)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"{subject_id}: failed to read {edf_path.name}: {exc}")
            continue
        print(f"{subject_id}: {edf_path.name} -> {hours:.2f} h")

    if not any_found:
        print(f"No Muse EDF files found under {base_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
