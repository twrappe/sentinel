"""CHB-MIT Scalp EEG dataset loader.

Dataset: https://physionet.org/content/chbmit/1.0.0/
Signals: EEG (scalp, 23 channels at 256 Hz)
Events:  Seizure onset / offset annotations
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sentinel.agents.eval_scoring_agent import GroundTruthLabel
from sentinel.fusion.signal_packager import BiosignalWindow

# Default root for raw data.  Override via SENTINEL_DATA_DIR env var or --data-dir flag.
_DEFAULT_DATA_DIR = Path("data/raw/chbmit")
_DEFAULT_PROCESSED_DIR = Path("data/processed/chbmit")
_DEFAULT_GROUND_TRUTH_DIR = Path("data/ground_truth/chbmit")


class CHBMITLoader:
    """Loads windowed EEG segments and seizure annotations from CHB-MIT.

    Usage::

        loader = CHBMITLoader(data_dir=Path("data/processed/chbmit"))
        for window, label in loader.iter_segments(subject="chb01"):
            ...
    """

    def __init__(
        self,
        data_dir: Path = _DEFAULT_PROCESSED_DIR,
        ground_truth_dir: Path = _DEFAULT_GROUND_TRUTH_DIR,
        window_seconds: float = 10.0,
        stride_seconds: float = 5.0,
    ) -> None:
        self.data_dir = data_dir
        self.ground_truth_dir = ground_truth_dir
        self.window_seconds = window_seconds
        self.stride_seconds = stride_seconds

    def iter_segments(
        self, subject: str | None = None
    ):  # -> Iterator[tuple[BiosignalWindow, GroundTruthLabel]]
        """Yield (BiosignalWindow, GroundTruthLabel) pairs for all available segments.

        Args:
            subject: Restrict to a single subject folder (e.g. ``"chb01"``).
                     If None, iterates all subjects.

        Raises:
            FileNotFoundError: If the processed data directory does not exist.
                               Run ``--preprocess`` first.
        """
        raise NotImplementedError(
            "CHBMITLoader.iter_segments is not yet implemented. "
            "Run `python -m sentinel.datasets.chbmit --download --preprocess` first."
        )


def _download(data_dir: Path) -> None:
    """Download CHB-MIT from PhysioNet via WFDB."""
    raise NotImplementedError(
        "Download not yet implemented. "
        "Manually download from https://physionet.org/content/chbmit/1.0.0/ "
        f"and place files in {data_dir}."
    )


def _preprocess(raw_dir: Path, processed_dir: Path, ground_truth_dir: Path) -> None:
    """Segment EDF files into windows and extract seizure annotations."""
    raise NotImplementedError("Preprocessing pipeline not yet implemented.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CHB-MIT dataset utilities")
    parser.add_argument("--download", action="store_true", help="Download raw dataset")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess into windows")
    parser.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA_DIR)
    args = parser.parse_args()

    if args.download:
        _download(args.data_dir)
    if args.preprocess:
        _preprocess(_DEFAULT_DATA_DIR, _DEFAULT_PROCESSED_DIR, _DEFAULT_GROUND_TRUTH_DIR)
