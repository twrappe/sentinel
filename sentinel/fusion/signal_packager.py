"""signal_packager — fuses EEG, EMG, and accelerometer windows into LLM-ready prompt context.

NOTE: The design of this module is an open problem flagged in the README.
The feature selection, inter-channel relationship description, and temporal
resolution to preserve all require domain expertise and physiological knowledge.
The implementation below provides a skeletal structure; key decisions marked
with TODO comments must be informed by clinical signal processing knowledge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class BiosignalWindow:
    """A fixed-length multi-modal biosignal window ready for packaging.

    Attributes:
        duration_seconds: Length of the window in seconds.
        sample_rate_hz: Shared sample rate across modalities (or the highest, if mixed).
        eeg_channels: Dict mapping channel name → numpy array of shape (n_samples,).
        emg_channels: Dict mapping channel name → numpy array of shape (n_samples,).
        accel_channels: Dict mapping axis label → numpy array of shape (n_samples,).
        metadata: Optional dict of additional context (subject ID, segment index, etc.).
    """

    duration_seconds: float
    sample_rate_hz: float
    eeg_channels: dict[str, np.ndarray] = field(default_factory=dict)
    emg_channels: dict[str, np.ndarray] = field(default_factory=dict)
    accel_channels: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def _summarise_channel(name: str, signal: np.ndarray, sample_rate: float) -> str:
    """Produce a one-line statistical summary for a single signal channel.

    TODO: Replace / extend with clinically meaningful features:
      - EEG: band power (delta, theta, alpha, beta, gamma), spectral edge frequency
      - EMG: RMS amplitude, zero-crossing rate, burst onset/offset
      - Accel: magnitude, dominant frequency, jerk
    """
    rms = float(np.sqrt(np.mean(signal**2)))
    peak = float(np.max(np.abs(signal)))
    mean = float(np.mean(signal))
    return f"  {name}: mean={mean:.4f}, rms={rms:.4f}, peak={peak:.4f}"


def package(window: BiosignalWindow) -> str:
    """Convert a ``BiosignalWindow`` into a structured text prompt for the LLM.

    The output is designed to convey clinically relevant signal characteristics
    without requiring the model to process raw numeric arrays.

    TODO: This function is the most domain-sensitive part of the pipeline.
    The feature representation, section ordering, and natural-language framing
    should be validated with someone who understands how EEG, EMG, and accel
    signals present physiologically during target event types.

    Args:
        window: The multi-modal biosignal window to package.

    Returns:
        A multi-section text block suitable for use as LLM user-message content.
    """
    lines: list[str] = [
        f"## Biosignal Segment",
        f"Duration: {window.duration_seconds:.2f}s | Sample rate: {window.sample_rate_hz:.0f} Hz",
    ]

    if window.metadata:
        meta_str = ", ".join(f"{k}={v}" for k, v in window.metadata.items())
        lines.append(f"Metadata: {meta_str}")

    if window.eeg_channels:
        lines.append("\n### EEG")
        for ch, sig in window.eeg_channels.items():
            lines.append(_summarise_channel(ch, sig, window.sample_rate_hz))

    if window.emg_channels:
        lines.append("\n### EMG")
        for ch, sig in window.emg_channels.items():
            lines.append(_summarise_channel(ch, sig, window.sample_rate_hz))

    if window.accel_channels:
        lines.append("\n### Accelerometer")
        for axis, sig in window.accel_channels.items():
            lines.append(_summarise_channel(axis, sig, window.sample_rate_hz))

    lines.append(
        "\nBased on the signal characteristics above, determine whether a clinically "
        "significant event is present in this segment. Respond with a JSON object."
    )

    return "\n".join(lines)
