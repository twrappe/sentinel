"""EEG Motor Movement/Imagery Dataset (EEGMMIDB) loader.

Dataset: https://physionet.org/content/eegmmidb/1.0.0/
Signals: EEG (64-channel, 160 Hz)
Events:  Motor imagery / movement task annotations
"""

from __future__ import annotations

from pathlib import Path

from sentinel.agents.eval_scoring_agent import GroundTruthLabel
from sentinel.fusion.signal_packager import BiosignalWindow

_DEFAULT_DATA_DIR = Path("data/raw/eegmmidb")
_DEFAULT_PROCESSED_DIR = Path("data/processed/eegmmidb")
_DEFAULT_GROUND_TRUTH_DIR = Path("data/ground_truth/eegmmidb")


class PhysioNetEEGMMILoader:
    """Loads windowed EEG segments and motor imagery annotations from EEGMMIDB.

    Usage::

        loader = PhysioNetEEGMMILoader()
        for window, label in loader.iter_segments():
            ...
    """

    def __init__(
        self,
        data_dir: Path = _DEFAULT_PROCESSED_DIR,
        ground_truth_dir: Path = _DEFAULT_GROUND_TRUTH_DIR,
        window_seconds: float = 4.0,
        stride_seconds: float = 2.0,
    ) -> None:
        self.data_dir = data_dir
        self.ground_truth_dir = ground_truth_dir
        self.window_seconds = window_seconds
        self.stride_seconds = stride_seconds

    def iter_segments(
        self, subject: str | None = None
    ):  # -> Iterator[tuple[BiosignalWindow, GroundTruthLabel]]
        """Yield (BiosignalWindow, GroundTruthLabel) pairs.

        Args:
            subject: Restrict to a single subject (e.g. ``"S001"``).
                     If None, iterates all subjects.

        Raises:
            NotImplementedError: Until preprocessing pipeline is implemented.
        """
        raise NotImplementedError(
            "PhysioNetEEGMMILoader.iter_segments is not yet implemented."
        )
