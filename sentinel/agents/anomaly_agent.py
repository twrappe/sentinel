"""AnomalyAgent — LLM call, structured output parsing, and confidence extraction."""

from __future__ import annotations

import json
from typing import Any

import anthropic
from pydantic import BaseModel, Field


class DetectedEvent(BaseModel):
    """A single anomaly event detected by the LLM.

    Fields cover four categories:
    - Detection outcome: ``detected``, ``event_type``
    - Temporal localisation: ``onset_seconds``, ``offset_seconds``,
      ``modality_onsets`` (per-modality, since EEG/EMG/accel events may not
      be simultaneous)
    - Confidence: ``confidence`` (overall), ``modality_confidence``
      (per-modality, so cross-modal disagreement is visible to the scorer)
    - Evidence quality: ``artifact_flag``, ``artifact_reason``,
      ``evidence_channels``, ``reasoning``
    """

    detected: bool = Field(description="Whether an anomalous event was detected.")
    event_type: str | None = Field(
        default=None,
        description="Classification of the detected event (e.g. 'seizure', 'motor_imagery').",
    )

    # --- Temporal localisation ---
    onset_seconds: float | None = Field(
        default=None,
        description=(
            "Estimated onset time of the event within the segment, in seconds. "
            "Use the earliest cross-modal onset when modalities disagree."
        ),
    )
    offset_seconds: float | None = Field(
        default=None,
        description="Estimated offset time of the event within the segment, in seconds.",
    )
    modality_onsets: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-modality onset times in seconds, keyed by modality name "
            "(e.g. {'eeg': 1.2, 'emg': 1.4}). Populate only for modalities "
            "where an onset can be independently identified."
        ),
    )

    # --- Confidence ---
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall model confidence in this detection (0–1).",
    )
    modality_confidence: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-modality confidence scores (0–1), keyed by modality name "
            "(e.g. {'eeg': 0.9, 'emg': 0.6, 'accel': 0.3}). Allows the "
            "scorer to detect cross-modal disagreement and weight evidence "
            "accordingly."
        ),
    )

    # --- Evidence quality ---
    artifact_flag: bool = Field(
        default=False,
        description=(
            "True if one or more signal channels are too noisy or artifacted "
            "to support a reliable detection decision. When True, 'detected' "
            "should be treated as low-confidence regardless of the confidence field."
        ),
    )
    artifact_reason: str | None = Field(
        default=None,
        description=(
            "Human-readable description of the artifact(s) observed "
            "(e.g. 'high-amplitude EMG contamination on EEG channels F3/C3'). "
            "Required when artifact_flag is True."
        ),
    )
    evidence_channels: list[str] = Field(
        default_factory=list,
        description="Signal channels cited as supporting evidence (e.g. ['EEG:F3', 'EMG:left']).",
    )
    reasoning: str = Field(
        default="", description="Free-text explanation of the detection decision."
    )


_SYSTEM_PROMPT = """\
You are a biosignal anomaly detection assistant. You will be given a multi-modal
biosignal segment represented as structured text. Your task is to determine whether
a clinically significant event is present.

Respond with a JSON object that conforms exactly to the DetectedEvent schema:
- Set `detected` to true only if you identify a clinically significant event.
- Provide `onset_seconds` and `offset_seconds` relative to the start of the segment.
- Populate `modality_onsets` with per-modality onset times where you can independently
  identify them — EEG, EMG, and accelerometer events may not be simultaneous.
- Set `confidence` as your overall certainty (0–1). Also populate `modality_confidence`
  for each modality present, so downstream scoring can detect cross-modal disagreement.
- If any channel is too noisy or artifacted to support a reliable decision, set
  `artifact_flag` to true and describe the artifact in `artifact_reason`.
- In `evidence_channels`, list every channel that supports your conclusion
  (e.g. 'EEG:F3', 'EMG:left_bicep').
- In `reasoning`, explain your decision concisely, noting any cross-modal agreement
  or disagreement that influenced your confidence."""

_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "detected": {"type": "boolean"},
        "event_type": {"type": ["string", "null"]},
        "onset_seconds": {"type": ["number", "null"]},
        "offset_seconds": {"type": ["number", "null"]},
        "modality_onsets": {
            "type": "object",
            "additionalProperties": {"type": "number"},
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "modality_confidence": {
            "type": "object",
            "additionalProperties": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "artifact_flag": {"type": "boolean"},
        "artifact_reason": {"type": ["string", "null"]},
        "evidence_channels": {"type": "array", "items": {"type": "string"}},
        "reasoning": {"type": "string"},
    },
    "required": ["detected", "confidence", "evidence_channels", "reasoning", "artifact_flag"],
}


class AnomalyAgent:
    """Wraps an Anthropic LLM call for biosignal anomaly detection."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic(api_key=api_key)

    def detect(self, prompt_context: str) -> DetectedEvent:
        """Run anomaly detection on a signal segment rendered as prompt context.

        Args:
            prompt_context: The text representation of the biosignal window,
                produced by ``signal_packager.package``.

        Returns:
            A ``DetectedEvent`` with detection outcome, timing, and confidence.
        """
        message = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt_context}],
        )

        raw_text = message.content[0].text  # type: ignore[index]
        payload = json.loads(raw_text)
        return DetectedEvent.model_validate(payload)
