"""Temporal precision — how accurately the model localises events in time."""

from __future__ import annotations

import numpy as np

from sentinel.agents.eval_scoring_agent import SegmentScore


def compute(
    scores: list[SegmentScore],
    tolerance_seconds: float = 2.0,
) -> dict[str, float]:
    """Compute temporal precision metrics over true-positive detections.

    NOTE: ``tolerance_seconds`` has a significant effect on this metric.
    The default (2s) is a placeholder. The appropriate value depends on
    annotation methodology for each dataset — see README for guidance.

    Args:
        scores: List of ``SegmentScore`` objects (only TPs with offset data are used).
        tolerance_seconds: Maximum offset to count as temporally correct.

    Returns:
        Dict with:
          - ``mean_offset_seconds``: Mean absolute temporal offset across TPs.
          - ``within_window_rate``: Fraction of TPs within ``tolerance_seconds``.
    """
    offsets = [
        s.temporal_offset_seconds
        for s in scores
        if s.true_positive and s.temporal_offset_seconds is not None
    ]

    if not offsets:
        return {"mean_offset_seconds": float("nan"), "within_window_rate": float("nan")}

    arr = np.array(offsets)
    return {
        "mean_offset_seconds": float(np.mean(arr)),
        "within_window_rate": float(np.mean(arr <= tolerance_seconds)),
    }
