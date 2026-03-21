"""Confidence calibration — do high-confidence outputs correlate with higher accuracy?"""

from __future__ import annotations

import numpy as np

from sentinel.agents.eval_scoring_agent import SegmentScore


def compute(
    scores: list[SegmentScore],
    n_bins: int = 10,
) -> dict[str, float]:
    """Compute Expected Calibration Error (ECE) and high-confidence accuracy.

    ECE measures the gap between average predicted confidence in each bin and
    the observed accuracy within that bin, weighted by bin size.

    Args:
        scores: List of per-segment ``SegmentScore`` objects.
        n_bins: Number of equal-width confidence bins for ECE calculation.

    Returns:
        Dict with:
          - ``ece``: Expected Calibration Error (lower is better).
          - ``high_confidence_accuracy``: Accuracy among segments where
            confidence ≥ 0.8.
    """
    if not scores:
        return {"ece": float("nan"), "high_confidence_accuracy": float("nan")}

    confidences = np.array([s.confidence for s in scores])
    is_correct = np.array([s.true_positive or (not s.false_positive and not s.false_negative) for s in scores], dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(scores)

    for low, high in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > low) & (confidences <= high)
        if not mask.any():
            continue
        bin_conf = float(np.mean(confidences[mask]))
        bin_acc = float(np.mean(is_correct[mask]))
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)

    high_conf_mask = confidences >= 0.8
    high_conf_acc = float(np.mean(is_correct[high_conf_mask])) if high_conf_mask.any() else float("nan")

    return {"ece": ece, "high_confidence_accuracy": high_conf_acc}
