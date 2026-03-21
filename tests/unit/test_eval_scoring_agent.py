"""Unit tests for EvalScoringAgent.

``EvalScoringAgent.score`` is the core comparison step in the eval pipeline:
it maps a (DetectedEvent, GroundTruthLabel) pair onto a ``SegmentScore`` with
TP/FP/FN flags, a temporal offset, and a channel-citation correctness flag.
These tests verify each output field in isolation using minimal fixtures.
"""

from sentinel.agents.anomaly_agent import DetectedEvent
from sentinel.agents.eval_scoring_agent import EvalScoringAgent, GroundTruthLabel


def _detection(detected=True, onset=1.0, channels=None, confidence=0.9):
    return DetectedEvent(
        detected=detected,
        event_type="seizure",
        onset_seconds=onset,
        offset_seconds=onset + 3.0 if onset is not None else None,
        confidence=confidence,
        evidence_channels=channels or ["EEG:F3", "EEG:C3"],
        reasoning="test",
    )


def _label(has_event=True, onset=1.0, channels=None):
    return GroundTruthLabel(
        has_event=has_event,
        event_type="seizure",
        onset_seconds=onset,
        channels=channels or ["EEG:F3"],
    )


class TestEvalScoringAgent:
    """Verifies that EvalScoringAgent correctly classifies detections and computes auxiliary scores.

    Each test exercises a single aspect of ``score()`` so that failures
    point unambiguously to the broken field. The agent is re-created per
    method via ``setup_method`` to prevent state leakage between tests.
    """

    def setup_method(self):
        # Use a 2s tolerance window — the same default used across the test suite.
        self.agent = EvalScoringAgent(temporal_tolerance_seconds=2.0)

    def test_true_positive(self):
        # Model detected an event that is present in the ground truth —
        # score must set true_positive and clear both false_positive and false_negative.
        score = self.agent.score(_detection(), _label())
        assert score.true_positive
        assert not score.false_positive
        assert not score.false_negative

    def test_false_positive(self):
        # Model fired on a clean segment (no ground-truth event) —
        # score must set false_positive and clear true_positive.
        score = self.agent.score(_detection(detected=True), _label(has_event=False))
        assert score.false_positive
        assert not score.true_positive

    def test_false_negative(self):
        # Model was silent on a segment with a real event —
        # score must set false_negative and clear true_positive.
        score = self.agent.score(_detection(detected=False), _label(has_event=True))
        assert score.false_negative
        assert not score.true_positive

    def test_temporal_offset_computed_for_tp(self):
        # For a TP with predicted onset 1.5s and ground-truth onset 1.0s,
        # the absolute offset must be computed as exactly 0.5s.
        score = self.agent.score(_detection(onset=1.5), _label(onset=1.0))
        assert score.temporal_offset_seconds is not None
        assert abs(score.temporal_offset_seconds - 0.5) < 1e-9

    def test_channel_citation_correct(self):
        # Detection cites the same channel as the ground truth label —
        # correct_channel_citation must be True.
        score = self.agent.score(
            _detection(channels=["EEG:F3"]), _label(channels=["EEG:F3"])
        )
        assert score.correct_channel_citation is True

    def test_channel_citation_incorrect(self):
        # Detection cites a channel not in the ground truth set —
        # correct_channel_citation must be False.
        score = self.agent.score(
            _detection(channels=["EEG:O1"]), _label(channels=["EEG:F3"])
        )
        assert score.correct_channel_citation is False

    def test_batch_length_mismatch_raises(self):
        # Passing unequal-length detection and label lists is a programmer error;
        # score_batch must raise ValueError rather than silently truncating.
        import pytest
        with pytest.raises(ValueError):
            self.agent.score_batch([_detection()], [_label(), _label()])
