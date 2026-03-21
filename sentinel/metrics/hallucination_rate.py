"""Hallucination rate — how often the model fires on clean or artifact-only segments.

NOTE: The current implementation is a single scalar (FP rate). The README
explicitly flags that a clinically meaningful hallucination *taxonomy* —
distinguishing failure types by physiological context (e.g. muscle artifact
vs. clean baseline) — requires domain expertise and cannot be defined here
without knowledge of how artifacts present across signal modalities.
"""

from __future__ import annotations

from sentinel.agents.eval_scoring_agent import SegmentScore


def compute(scores: list[SegmentScore]) -> float:
    """Compute the hallucination (false positive) rate.

    Rate = FP / (FP + TN)

    Args:
        scores: List of per-segment ``SegmentScore`` objects.

    Returns:
        Hallucination rate in [0, 1], or 0.0 if there are no negative ground
        truth segments.
    """
    fp = sum(1 for s in scores if s.false_positive)
    tn = sum(1 for s in scores if not s.true_positive and not s.false_negative and not s.false_positive)
    denominator = fp + tn
    return fp / denominator if denominator > 0 else 0.0
