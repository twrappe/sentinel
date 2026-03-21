"""Unit tests for eval metrics.

Each test class covers one metric module. Tests use ``_make_score`` to build
``SegmentScore`` fixtures with only the fields relevant to the metric under test;
all other fields default to neutral / falsy values.
"""

from sentinel.agents.eval_scoring_agent import SegmentScore
from sentinel.metrics import (
    confidence_calibration,
    detection_recall,
    hallucination_rate,
    temporal_precision,
)


def _make_score(tp=False, fp=False, fn=False, offset=None, channel_ok=None, confidence=0.8):
    return SegmentScore(
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        temporal_offset_seconds=offset,
        correct_channel_citation=channel_ok,
        confidence=confidence,
    )


class TestDetectionRecall:
    """Verifies detection_recall.compute correctly implements TP / (TP + FN).

    Detection recall measures the fraction of ground-truth events that the
    model successfully detected. It is insensitive to false positives, so
    hallucination behaviour does not influence this metric.
    """

    def test_perfect_recall(self):
        # All segments are true positives — recall must be 1.0 (no missed events).
        scores = [_make_score(tp=True), _make_score(tp=True)]
        assert detection_recall.compute(scores) == 1.0

    def test_zero_recall(self):
        # All segments are false negatives — the model missed every event, so recall is 0.0.
        scores = [_make_score(fn=True), _make_score(fn=True)]
        assert detection_recall.compute(scores) == 0.0

    def test_partial_recall(self):
        # One TP and one FN — recall should be exactly 0.5.
        scores = [_make_score(tp=True), _make_score(fn=True)]
        assert detection_recall.compute(scores) == 0.5

    def test_no_positives_returns_zero(self):
        # When there are no ground-truth positive segments (only FPs), the
        # denominator is zero and the function must return 0.0 rather than raise.
        scores = [_make_score(fp=True)]
        assert detection_recall.compute(scores) == 0.0


class TestTemporalPrecision:
    """Verifies temporal_precision.compute correctly scores event localisation.

    Temporal precision measures how accurately the model pinpoints the onset of
    a detected event. Only true-positive segments with onset offset data
    contribute; the metric is parameterised by a configurable tolerance window.
    """

    def test_all_within_window(self):
        # Both TPs have offsets (0.5s and 1.0s) that fall inside the 2s tolerance
        # window — within_window_rate must be 1.0.
        scores = [_make_score(tp=True, offset=0.5), _make_score(tp=True, offset=1.0)]
        result = temporal_precision.compute(scores, tolerance_seconds=2.0)
        assert result["within_window_rate"] == 1.0

    def test_none_within_window(self):
        # The single TP has a 5s offset, which exceeds the 2s tolerance window —
        # within_window_rate must be 0.0.
        scores = [_make_score(tp=True, offset=5.0)]
        result = temporal_precision.compute(scores, tolerance_seconds=2.0)
        assert result["within_window_rate"] == 0.0

    def test_no_tp_offset_data_returns_nan(self):
        # When there are no true positives with temporal offset data (e.g. only FPs),
        # the metric cannot be computed and must return NaN for both output fields.
        scores = [_make_score(fp=True)]
        result = temporal_precision.compute(scores)
        assert result["mean_offset_seconds"] != result["mean_offset_seconds"]  # NaN check


class TestHallucinationRate:
    """Verifies hallucination_rate.compute correctly implements FP / (FP + TN).

    Hallucination rate measures how often the model fires on segments that
    contain no real event. A high rate indicates the model is over-triggering
    on clean baselines or artefact-only segments.
    """

    def test_all_hallucinations(self):
        # Every segment is a false positive — the model hallucinated on all
        # clean segments, so the rate must be 1.0.
        scores = [_make_score(fp=True), _make_score(fp=True)]
        assert hallucination_rate.compute(scores) == 1.0

    def test_no_hallucinations(self):
        # A single true negative (no event present, model correctly silent) —
        # hallucination rate must be 0.0.
        tn = SegmentScore(
            true_positive=False,
            false_positive=False,
            false_negative=False,
            temporal_offset_seconds=None,
            correct_channel_citation=None,
            confidence=0.1,
        )
        assert hallucination_rate.compute([tn]) == 0.0


class TestConfidenceCalibration:
    """Verifies confidence_calibration.compute produces a well-formed result dict.

    Confidence calibration measures whether the model's stated confidence scores
    are meaningful — i.e. that high-confidence outputs have genuinely higher
    accuracy than low-confidence ones. ECE (Expected Calibration Error) quantifies
    this gap; lower is better.
    """

    def test_returns_expected_keys(self):
        # The output dict must always contain both 'ece' and
        # 'high_confidence_accuracy', regardless of score composition.
        scores = [_make_score(tp=True, confidence=0.9), _make_score(fp=True, confidence=0.3)]
        result = confidence_calibration.compute(scores)
        assert "ece" in result
        assert "high_confidence_accuracy" in result

    def test_empty_scores_returns_nan(self):
        # With no scored segments there is no calibration data — both output
        # fields must be NaN rather than raising an exception.
        result = confidence_calibration.compute([])
        assert result["ece"] != result["ece"]  # NaN
