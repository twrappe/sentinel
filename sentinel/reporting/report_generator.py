"""report_generator — aggregates eval metrics into structured JSON and a human-readable summary."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sentinel.agents.eval_scoring_agent import SegmentScore
from sentinel.metrics import (
    confidence_calibration,
    detection_recall,
    hallucination_rate,
    temporal_precision,
)


def build_report(
    *,
    run_id: str,
    model: str,
    dataset: str,
    scores: list[SegmentScore],
    temporal_tolerance_seconds: float = 2.0,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate per-segment scores into a structured eval report dict.

    Args:
        run_id: Unique identifier for this eval run.
        model: Model name used for detection.
        dataset: Dataset name evaluated against.
        scores: All per-segment ``SegmentScore`` objects for the run.
        temporal_tolerance_seconds: Passed through to temporal precision metric.
        extra: Optional additional fields to include in the report.

    Returns:
        Dict matching the output format described in the README.
    """
    recall = detection_recall.compute(scores)
    temp = temporal_precision.compute(scores, tolerance_seconds=temporal_tolerance_seconds)
    halluc = hallucination_rate.compute(scores)
    calib = confidence_calibration.compute(scores)

    channel_citations = [
        s.correct_channel_citation
        for s in scores
        if s.true_positive and s.correct_channel_citation is not None
    ]
    channel_citation_rate = (
        sum(channel_citations) / len(channel_citations) if channel_citations else float("nan")
    )

    report: dict[str, Any] = {
        "run_id": run_id,
        "model": model,
        "dataset": dataset,
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "segments_evaluated": len(scores),
        "metrics": {
            "detection_recall": recall,
            "temporal_precision": {
                "mean_offset_seconds": temp["mean_offset_seconds"],
                f"within_{temporal_tolerance_seconds}s_window": temp["within_window_rate"],
            },
            "hallucination_rate": halluc,
            "confidence_calibration": {
                "ece": calib["ece"],
                "high_confidence_accuracy": calib["high_confidence_accuracy"],
            },
        },
        "cross_modal_faithfulness": {
            "correct_channel_citation_rate": channel_citation_rate,
        },
    }

    if extra:
        report.update(extra)

    return report


def save_report(report: dict[str, Any], output_dir: Path) -> Path:
    """Write the report as ``report.json`` inside ``output_dir``.

    Args:
        report: Dict produced by ``build_report``.
        output_dir: Directory to write the report into (created if absent).

    Returns:
        Path to the written file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    return out_path


def format_summary(report: dict[str, Any]) -> str:
    """Render a human-readable text summary of the report.

    Args:
        report: Dict produced by ``build_report``.

    Returns:
        Multi-line string suitable for printing to stdout.
    """
    m = report["metrics"]
    tp = m["temporal_precision"]
    calib = m["confidence_calibration"]
    cf = report["cross_modal_faithfulness"]

    within_key = next((k for k in tp if k.startswith("within_")), None)
    within_val = f"{tp[within_key]:.2%}" if within_key and tp[within_key] == tp[within_key] else "n/a"

    lines = [
        f"=== SENTINEL Eval Report ===",
        f"Run:      {report['run_id']}",
        f"Model:    {report['model']}",
        f"Dataset:  {report['dataset']}",
        f"Segments: {report['segments_evaluated']}",
        f"",
        f"Detection Recall:           {m['detection_recall']:.2%}",
        f"Hallucination Rate:         {m['hallucination_rate']:.2%}",
        f"Temporal Mean Offset:       {tp['mean_offset_seconds']:.2f}s",
        f"Temporal Within-Window:     {within_val}",
        f"Channel Citation Rate:      {cf['correct_channel_citation_rate']:.2%}",
        f"ECE:                        {calib['ece']:.4f}",
        f"High-Confidence Accuracy:   {calib['high_confidence_accuracy']:.2%}",
    ]
    return "\n".join(lines)
