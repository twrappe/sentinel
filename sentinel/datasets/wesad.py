"""WESAD (Wearable Stress and Affect Detection) dataset loader.

Dataset: https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection
Signals: EEG (chest + wrist BVP, EDA, EMG, Temp, Accel)
Events:  Stress, amusement, and baseline affect states

Note: WESAD includes chest-worn (RespiBAN) and wrist-worn (Empatica E4) devices.
The fusion packager must account for differing sample rates across modalities.
"""

from __future__ import annotations

from pathlib import Path

from sentinel.agents.eval_scoring_agent import GroundTruthLabel
from sentinel.fusion.signal_packager import BiosignalWindow

_DEFAULT_DATA_DIR = Path("data/raw/wesad")
_DEFAULT_PROCESSED_DIR = Path("data/processed/wesad")
_DEFAULT_GROUND_TRUTH_DIR = Path("data/ground_truth/wesad")

# WESAD affect labels
LABEL_BASELINE = 1
LABEL_STRESS = 2
LABEL_AMUSEMENT = 3
LABEL_MEDITATION = 4


class WESADLoader:
    """Loads multi-modal segments and affect-state annotations from WESAD.

    Usage::

        loader = WESADLoader()
        for window, label in loader.iter_segments():
            ...
    """

    def __init__(
        self,
        data_dir: Path = _DEFAULT_PROCESSED_DIR,
        ground_truth_dir: Path = _DEFAULT_GROUND_TRUTH_DIR,
        window_seconds: float = 60.0,
        stride_seconds: float = 30.0,
        target_labels: list[int] | None = None,
    ) -> None:
        self.data_dir = data_dir
        self.ground_truth_dir = ground_truth_dir
        self.window_seconds = window_seconds
        self.stride_seconds = stride_seconds
        # Default: distinguish stress vs rest
        self.target_labels = target_labels or [LABEL_STRESS]

    def iter_segments(
        self, subject: str | None = None
    ):  # -> Iterator[tuple[BiosignalWindow, GroundTruthLabel]]
        """Yield (BiosignalWindow, GroundTruthLabel) pairs.

        The BiosignalWindow will include EMG and accel channels from the
        appropriate device. EEG is not present in WESAD; chest EDA and BVP
        are mapped to emg_channels and eeg_channels keys respectively as
        a practical approximation until the fusion layer is fully designed.

        TODO: Finalise the signal-to-slot mapping in collaboration with the
        signal_packager design — see README for guidance.

        Args:
            subject: Restrict to a single subject (e.g. ``"S2"``).
                     If None, iterates all subjects.

        Raises:
            NotImplementedError: Until preprocessing pipeline is implemented.
        """
        raise NotImplementedError("WESADLoader.iter_segments is not yet implemented.")
