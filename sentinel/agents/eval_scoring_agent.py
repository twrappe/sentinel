"""EvalScoringAgent — scores AnomalyAgent detections against ground truth labels."""

from __future__ import annotations

from dataclasses import dataclass, field

from sentinel.agents.anomaly_agent import DetectedEvent


@dataclass
class GroundTruthLabel:
    """A single annotated event label for a biosignal segment."""

    has_event: bool
    event_type: str | None = None
    onset_seconds: float | None = None
    offset_seconds: float | None = None
    channels: list[str] = field(default_factory=list)


@dataclass
class SegmentScore:
    """Scoring result for a single segment comparison."""

    true_positive: bool
    false_positive: bool
    false_negative: bool
    temporal_offset_seconds: float | None
    correct_channel_citation: bool | None
    confidence: float


class EvalScoringAgent:
    """Compares ``DetectedEvent`` outputs to ``GroundTruthLabel`` annotations.

    Args:
        temporal_tolerance_seconds: Maximum allowed offset between predicted and
            ground-truth onset to count as a correct temporal localisation.
            NOTE: The appropriate value varies by dataset and annotation type —
            see README for guidance.
    """

    def __init__(self, temporal_tolerance_seconds: float = 2.0) -> None:
        self.temporal_tolerance_seconds = temporal_tolerance_seconds

    def score(
        self, detection: DetectedEvent, ground_truth: GroundTruthLabel
    ) -> SegmentScore:
        """Score a single detection against its ground truth label.

        Args:
            detection: Output from ``AnomalyAgent.detect``.
            ground_truth: The annotated label for the same segment.

        Returns:
            A ``SegmentScore`` with TP/FP/FN flags and auxiliary metrics.
        """
        tp = detection.detected and ground_truth.has_event
        fp = detection.detected and not ground_truth.has_event
        fn = not detection.detected and ground_truth.has_event

        temporal_offset: float | None = None
        if tp and detection.onset_seconds is not None and ground_truth.onset_seconds is not None:
            temporal_offset = abs(detection.onset_seconds - ground_truth.onset_seconds)

        channel_correct: bool | None = None
        if tp and ground_truth.channels:
            cited = set(detection.evidence_channels)
            expected = set(ground_truth.channels)
            channel_correct = bool(cited & expected)  # at least one correct channel cited

        return SegmentScore(
            true_positive=tp,
            false_positive=fp,
            false_negative=fn,
            temporal_offset_seconds=temporal_offset,
            correct_channel_citation=channel_correct,
            confidence=detection.confidence,
        )

    def score_batch(
        self,
        detections: list[DetectedEvent],
        ground_truths: list[GroundTruthLabel],
    ) -> list[SegmentScore]:
        """Score a list of detections against their paired ground truth labels."""
        if len(detections) != len(ground_truths):
            raise ValueError(
                f"Mismatch: {len(detections)} detections vs {len(ground_truths)} labels."
            )
        return [self.score(d, g) for d, g in zip(detections, ground_truths)]
