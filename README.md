# SENTINEL
### Biosignal Event and Anomaly Evaluation Framework

> *Systematic evaluation of LLM-based anomaly detection over multi-modal biosignal data.*

---

## Overview

SENTINEL is an LLM evaluation framework designed to measure how reliably a language model detects clinically significant events from fused multi-modal biosignal inputs — EEG, EMG, and accelerometer.

Where most LLM eval frameworks operate on text-in / text-out tasks, SENTINEL addresses a harder problem: **evaluating AI judgment on inherently ambiguous, temporally-structured physiological data.** It measures not just whether the model detected an event, but *when*, *why*, and *how confident it should have been*.

The framework is intentionally domain-agnostic at the architecture level and biosignal-specific at the metric level — making it extensible to new signal types while remaining clinically meaningful out of the box.

---

## Motivation

The research literature correlating biosignal modalities to clinical outcomes is mature and well-evidenced. The translation of that evidence into accessible, clinically validated products, however, remains limited.

The primary bottleneck is not scientific. For neurological conditions characterised by symptoms that are sudden, episodic, and difficult to observe in controlled clinical settings — seizures, psychotic episodes, manic states — the pathway from a validated biosignal biomarker to a regulatory-approved, patient-accessible device is extensive and demanding. Regulatory frameworks governing medical devices, particularly those indicated for neurological conditions, require a level of demonstrated reliability and safety that most academic research pipelines are not structured to produce.

AI systems applied to wearable and ambulatory biosignal data encounter this same constraint at the evaluation layer. Most LLM evaluation tooling assesses health AI using the same methods applied to general-purpose language models — exact match, BLEU score, and human preference ratings — without accounting for the clinical significance of errors. This approach is insufficient when model outputs constitute, or directly inform, clinical decisions.

SENTINEL was developed on the premise that **AI quality engineering for health sensors requires domain-specific evaluation architecture** — infrastructure with metrics that account for temporal proximity, cross-modal evidence, and the distinction between a clinically significant hallucination and a benign false positive. The objective is not only to quantify model performance, but to do so in terms that are meaningful within the regulatory and clinical contexts where these systems are intended to operate.

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

Three agents, one pipeline. Intentionally mirrors the multi-agent pattern from [PRISM](https://github.com/twrappe/prism) — the same orchestration principles apply whether you're triaging CI/CD failures or evaluating biosignal AI.

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
  }
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

## Roadmap

- [ ] CHB-MIT dataset loader + seizure eval baseline
- [ ] WESAD multi-modal fusion packager
- [ ] Confidence calibration plots (ECE curves)
- [ ] Support for open-weight models via Ollama
- [ ] Streaming eval mode for real-time sensor feeds

---

## Background

This project draws on prior work in EEG/BCI research and biosignal quality engineering. The eval methodology is informed by both clinical signal processing literature and modern LLM evaluation practices (DeepEval, Ragas).

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

## License

MIT
