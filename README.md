# SENTINEL
### Biosignal Event and Anomaly Evaluation Framework

> *Systematic evaluation of LLM-based anomaly detection over multi-modal biosignal data.*

---

## Overview

Neurological AI fails quietly. A model that detects seizures with 84% recall sounds impressive until you consider what happens in the other 16% — and whether the model had any idea it was about to miss. Most LLM evaluation frameworks cannot answer that question. They were built for text tasks, and they measure what is easy to measure: whether the answer matched, whether a human preferred it. They have no concept of temporal proximity, cross-modal evidence, or the difference between a hallucination on a clean signal and one triggered by a muscle artifact.

SENTINEL addresses this by treating biosignal evaluation as a first-class engineering problem. It fuses multi-modal physiological data — EEG, EMG, accelerometer — into structured prompt context, runs it through an LLM detection pipeline, and scores the output against ground truth using metrics designed for the failure modes that actually matter in clinical environments: not just whether an event was detected, but when, why, and how confident the model should have been.

The result is a domain-specific eval framework that measures five things general-purpose tools cannot: detection recall, temporal precision, cross-modal faithfulness, hallucination rate, and confidence calibration. It runs against three public datasets — CHB-MIT, EEGMMIDB, and WESAD — and produces structured JSON reports suitable for both research analysis and regulatory documentation. The architecture is intentionally extensible: biosignal-specific at the metric level, domain-agnostic at the pipeline level.

---

## Motivation

The bottleneck is not scientific. For conditions characterized by symptoms that are sudden, episodic, and difficult to observe in controlled settings, the pathway from a validated biosignal biomarker to a patient-accessible device is long and demanding. Regulatory frameworks for neurological AI require a level of demonstrated reliability that most academic research pipelines are not structured to produce — and most AI evaluation tooling is not designed to support.

The core problem is metric mismatch. Most LLM evaluation frameworks assess health AI the same way they assess a chatbot — exact match, BLEU score, human preference ratings. These metrics are blind to the clinical significance of errors. They cannot distinguish a hallucinated event on a clean baseline segment from one triggered by a muscle artifact. They cannot penalize a correct detection that arrived four seconds too late. They have no concept of what a miscalibrated confidence score costs when it informs a clinical decision at 2am.

SENTINEL was built on the premise that AI quality engineering for health sensors requires domain-specific evaluation architecture — infrastructure with metrics that account for temporal proximity, cross-modal evidence, and the difference between a clinically significant hallucination and a benign false positive. The objective is not only to quantify model performance, but to do so in terms that are meaningful in the regulatory and clinical contexts where these systems will eventually operate.

---

## Why Now

Large language models have made a specific capability newly accessible: the ability to reason over complex, multi-modal inputs and return structured, interpretable outputs without task-specific supervised training. For biosignal analysis, this matters. The previous generation of neurological event detection systems required labeled datasets large enough to train domain-specific classifiers from scratch — a bottleneck that kept the technology inside research institutions. LLMs lower that barrier significantly.

But capability without trust infrastructure is not a product. As LLMs move into clinical adjacent applications, the absence of domain-appropriate evaluation tooling is becoming the binding constraint. General-purpose benchmarks were not designed for systems where a false positive has a physiological context and a missed detection has a consequence. The window to establish evaluation standards for this domain — before the field consolidates around inadequate ones — is open now and will not stay open indefinitely.

SENTINEL is built for this moment.

---

## Architecture

```
biosignal segments
       │
       ▼
┌─────────────────────┐
│   signal_packager   │  ← fuses EEG + EMG + accel into LLM-ready prompt context
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   AnomalyAgent      │  ← LLM call, structured output parsing, confidence extraction
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  EvalScoringAgent   │  ← compares detections against ground truth labels
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  report_generator   │  ← aggregates metrics → structured JSON + human-readable summary
└─────────────────────┘
```

Three agents, one pipeline. Separating detection from scoring makes the eval layer independently auditable — the same orchestration principles apply whether you're triaging CI/CD failures or evaluating biosignal AI. Intentionally mirrors the multi-agent pattern from [PRISM](https://github.com/twrappe/prism).

---

## Eval Metrics

| Metric | What It Measures |
|---|---|
| **Detection Recall** | Did the model catch the event at all? |
| **Temporal Precision** | How accurately did it localize the event in time? |
| **Cross-Modal Faithfulness** | Did it cite the correct signal channels as evidence? |
| **Hallucination Rate** | How often does it flag events on clean or artifact-only segments? |
| **Confidence Calibration** | Do high-confidence outputs actually correlate with higher accuracy? |

Temporal precision uses a configurable tolerance window rather than exact match — because clinical annotations themselves carry inherent boundary ambiguity.

---

## Supported Datasets

| Dataset | Signals | Events |
|---|---|---|
| [CHB-MIT Scalp EEG](https://physionet.org/content/chbmit/1.0.0/) | EEG | Seizure onset/offset |
| [EEGMMIDB](https://physionet.org/content/eegmmidb/1.0.0/) | EEG | Motor imagery events |
| [WESAD](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection) | EEG, ECG, EMG, Accel | Stress / affect states |

All datasets are publicly available and require no clinical data access agreements.

---

## Project Structure

```
sentinel/
├── sentinel/
│   ├── agents/
│   │   ├── anomaly_agent.py          # LLM call + structured output parsing
│   │   └── eval_scoring_agent.py     # Scores detections vs ground truth
│   ├── fusion/
│   │   └── signal_packager.py        # Multi-modal → LLM prompt context
│   ├── metrics/
│   │   ├── detection_recall.py
│   │   ├── temporal_precision.py
│   │   ├── hallucination_rate.py
│   │   └── confidence_calibration.py
│   ├── datasets/
│   │   ├── chbmit.py
│   │   ├── physionet.py
│   │   └── wesad.py
│   └── reporting/
│       └── report_generator.py
├── data/
│   ├── raw/                          # Downloaded datasets (gitignored)
│   ├── processed/                    # Windowed, normalized segments
│   └── ground_truth/                 # Labeled annotations per dataset
├── runs/
│   └── results/                      # Eval run outputs (gitignored)
├── tests/
│   ├── unit/
│   └── integration/
├── docker-compose.yml
└── pyproject.toml
```

---

## Quickstart

```bash
git clone https://github.com/twrappe/sentinel
cd sentinel
pip install -e .

# Download and preprocess a dataset
python -m sentinel.datasets.chbmit --download --preprocess

# Run an eval campaign
python -m sentinel.run --dataset chbmit --model claude-sonnet-4-20250514 --output runs/results/

# View results
cat runs/results/latest/report.json
```

---

## Output Format

Each eval run produces a structured JSON report:

```json
{
  "run_id": "chbmit_20260321_001",
  "model": "claude-sonnet-4-20250514",
  "dataset": "chbmit",
  "segments_evaluated": 500,
  "metrics": {
    "detection_recall": 0.84,
    "temporal_precision": {
      "mean_offset_seconds": 1.2,
      "within_2s_window": 0.91
    },
    "hallucination_rate": 0.06,
    "confidence_calibration": {
      "ece": 0.043,
      "high_confidence_accuracy": 0.93
    }
  },
  "cross_modal_faithfulness": {
    "correct_channel_citation_rate": 0.78
  },
  "gate_status": "PASS"
}
```

The following example shows a run where SENTINEL flagged meaningful quality degradation — high hallucination rate on artifact-heavy segments and temporal precision collapse on seizure boundary cases:

```json
{
  "run_id": "chbmit_20260321_002",
  "model": "gpt-4o",
  "dataset": "chbmit",
  "segments_evaluated": 500,
  "metrics": {
    "detection_recall": 0.81,
    "temporal_precision": {
      "mean_offset_seconds": 4.7,
      "within_2s_window": 0.43
    },
    "hallucination_rate": 0.31,
    "confidence_calibration": {
      "ece": 0.19,
      "high_confidence_accuracy": 0.61
    }
  },
  "cross_modal_faithfulness": {
    "correct_channel_citation_rate": 0.52
  },
  "gate_status": "FAIL",
  "failure_reasons": [
    "hallucination_rate exceeds threshold (0.31 > 0.10)",
    "temporal_precision.within_2s_window below threshold (0.43 < 0.80)",
    "confidence_calibration.ece exceeds threshold (0.19 > 0.10)"
  ],
  "diagnostic_note": "Elevated hallucination rate concentrated on EMG artifact segments. Temporal precision collapse suggests boundary ambiguity in seizure onset annotations for patients 7 and 12. High-confidence outputs poorly calibrated — model is asserting certainty it has not earned."
}
```

---

## Relation to PRISM

SENTINEL shares architectural DNA with [PRISM](https://github.com/twrappe/prism), a multi-agent RCA system for CI/CD pipelines in silicon validation environments. Both systems:

- Use an LLM agent to detect and classify events in structured input streams
- Output confidence-scored JSON with supporting evidence
- Separate the detection layer from the evaluation/scoring layer
- Are designed for high-volume, async processing of heterogeneous inputs

SENTINEL extends this pattern into the biosignal domain, where ground truth ambiguity, temporal scoring, and cross-modal evidence attribution introduce eval challenges not present in software pipeline analysis.

---

## Current Status

### Complete
- Multi-agent pipeline architecture (signal_packager → AnomalyAgent → EvalScoringAgent → report_generator)
- Core metric implementations: detection recall, temporal precision, hallucination rate, confidence calibration (ECE)
- CHB-MIT dataset loader
- Structured JSON report output with PASS/FAIL gating
- Docker Compose configuration

### In Progress
- CHB-MIT end-to-end eval baseline (first full run against real data)
- WESAD multi-modal fusion packager (EMG + accel integration)
- Integration tests covering the full detect → score → report path

### Planned

**Signal Feature Quality**
- Replace mean/RMS/peak summaries with clinically meaningful features
  - EEG: band power (delta, theta, alpha, beta, gamma), spectral edge frequency
  - EMG: RMS amplitude, zero-crossing rate, burst onset/offset
  - Accel: magnitude, dominant frequency, jerk
- Validate feature representation with domain expert or clinical literature
- Benchmark LLM detection performance before and after feature improvement

**Expanded Metrics**
- PPV / NPV (positive and negative predictive value)
- Explicit specificity alongside hallucination rate
- F-beta score (configurable to weight misses vs. false alarms per use case)
- Artifact flag accuracy scoring
- Per-subject stratification — surface whether aggregate metrics hide subject-level variance
- Severity-weighted recall — penalise missing longer events more than shorter ones
- LLM consistency metric — same segment twice should produce the same detection

**Extended Biosignal Coverage**
- EEGMMIDB loader implementation
- Additional signal modalities: ECG, EDA, skin temperature (already present in WESAD)
- Cross-dataset generalisation evaluation (CHB-MIT vs. EEGMMIDB delta)
- Additional public datasets: sleep staging, Parkinson's tremor
- Hallucination taxonomy by physiological context (artifact-triggered vs. clean-baseline false positives)

**Operationalisation**
- Streaming eval mode for real-time sensor feeds
- Confidence calibration plots (ECE curves)
- Support for open-weight models via Ollama
- Cost and latency tracking per segment
- Human-readable report translation layer for clinical audiences

---

## Background

SENTINEL builds on professional and research experience at the intersection of biosignal engineering and clinical AI, including signal processing, hardware validation, and quality engineering for wearable physiological data pipelines. The evaluation methodology draws on modern LLM eval practices — DeepEval, Ragas — extended to meet the demands of a clinical signal domain.

---

## Challenges for Human Developer

Several components of SENTINEL require domain expertise that cannot be derived from documentation or general software engineering practice alone. The following areas are explicitly flagged for human judgment.

**Signal Packager Design**
The `fusion/signal_packager.py` module has no established pattern to follow. Representing a multi-channel biosignal window — EEG frequency bands, EMG amplitude profiles, accelerometer vectors — as LLM-ready prompt context in a way that preserves clinically relevant structure is an open design problem. Decisions about which features to surface, how to describe inter-channel relationships, and what temporal resolution to preserve require firsthand understanding of how these signals behave physiologically.

**Ground Truth Tolerance Windows**
The temporal precision metric uses a configurable tolerance window when scoring detection offset. The default placeholder value is arbitrary. Setting it meaningfully requires knowing how much boundary ambiguity is inherent in each annotation type — seizure onsets, motor imagery events, and stress state transitions each carry different degrees of clinical uncertainty. These values should be set by someone who understands the labeling methodology of each dataset, not inferred from the data itself.

**Hallucination Taxonomy**
The current hallucination rate metric is a single scalar. In practice, not all false positives are equivalent — flagging an event on a muscle artifact segment reflects a different failure mode than flagging on a clean baseline segment. A clinically meaningful hallucination taxonomy, distinguishing failure types by their physiological context, would significantly increase the diagnostic value of eval results. This taxonomy cannot be defined without domain knowledge of how artifacts present across signal modalities.

**Dataset Annotation Edge Cases**
The CHB-MIT and WESAD datasets contain labeling inconsistencies, overlapping annotations, and ambiguous boundary cases. Decisions about how to handle these — exclude them from eval, include them with reduced weight, or create a separate ambiguous-label tier — will materially affect all reported metrics. These judgment calls require understanding of the underlying physiology and the original annotation methodology, not just the data schema.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

Copyright 2026 Thomas Wrappe

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for the full license text, or visit http://www.apache.org/licenses/LICENSE-2.0.
