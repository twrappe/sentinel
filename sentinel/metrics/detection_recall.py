"""Detection recall — proportion of ground truth events caught by the model."""

from __future__ import annotations

from sentinel.agents.eval_scoring_agent import SegmentScore


def compute(scores: list[SegmentScore]) -> float:
    """Compute detection recall over a set of scored segments.

    Recall = TP / (TP + FN)

    Args:
        scores: List of per-segment ``SegmentScore`` objects.

    Returns:
        Recall in [0, 1], or 0.0 if there are no positive ground truth events.
    """
    tp = sum(1 for s in scores if s.true_positive)
    fn = sum(1 for s in scores if s.false_negative)
    denominator = tp + fn
    return tp / denominator if denominator > 0 else 0.0
