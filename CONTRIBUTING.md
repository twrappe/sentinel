# Contributing to SENTINEL

Thanks for your interest in contributing. SENTINEL is an early-stage research framework — contributions that improve evaluation rigor, signal feature quality, or dataset coverage are especially welcome.

---

## Where to Start

Check the [Current Status](README.md#current-status) section of the README for what's actively in progress and what's planned. Open issues are the best place to find work that's been scoped and is ready to pick up.

If you have an idea that isn't tracked yet, open an issue before writing code. This keeps effort from being duplicated and ensures contributions align with the project direction.

---

## Areas Most Needing Contribution

Several components of SENTINEL are explicitly flagged as requiring domain expertise that the core maintainer cannot fully supply alone:

- **Signal packager design** — how to represent multi-channel biosignal windows as LLM-ready prompt context in a clinically meaningful way
- **Ground truth tolerance windows** — setting annotation boundary tolerances per dataset based on labeling methodology
- **Hallucination taxonomy** — defining failure mode categories by physiological context (artifact-triggered vs. clean-baseline false positives)
- **Dataset annotation edge cases** — judgment calls on overlapping or ambiguous labels in CHB-MIT and WESAD

If you have a background in neuroscience, clinical AI, or biosignal processing, these are the highest-leverage places to contribute.

---

## Contribution Process

1. Fork the repository and create a branch from `main`
2. Make your changes with tests where applicable
3. Ensure existing tests pass: `pytest tests/`
4. Open a pull request with a clear description of what changed and why

For larger changes, open an issue first to discuss the approach.

---

## Code Style

- Python 3.10+
- Format with `black`, lint with `ruff`
- Type hints on all public functions
- Docstrings on modules and non-trivial functions

---

## Reporting Issues

Use GitHub Issues. For bugs, include the dataset, model, and run configuration. For metric or scoring questions, include the relevant segment context and ground truth label if possible.

---

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
