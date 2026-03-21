"""SENTINEL eval runner — entry point for ``python -m sentinel.run``."""

from __future__ import annotations

import uuid
from pathlib import Path

import typer
from rich.console import Console

from sentinel.agents.anomaly_agent import AnomalyAgent
from sentinel.agents.eval_scoring_agent import EvalScoringAgent
from sentinel.reporting import report_generator

app = typer.Typer(help="Run a SENTINEL biosignal eval campaign.")
console = Console()

_DATASET_LOADERS = {
    "chbmit": "sentinel.datasets.chbmit.CHBMITLoader",
    "eegmmidb": "sentinel.datasets.physionet.PhysioNetEEGMMILoader",
    "wesad": "sentinel.datasets.wesad.WESADLoader",
}


@app.command()
def run(
    dataset: str = typer.Option(..., help=f"Dataset to evaluate. One of: {list(_DATASET_LOADERS)}"),
    model: str = typer.Option("claude-sonnet-4-20250514", help="Anthropic model to use."),
    output: Path = typer.Option(Path("runs/results/"), help="Directory to write results."),
    subject: str | None = typer.Option(None, help="Restrict to a single subject ID."),
    temporal_tolerance: float = typer.Option(2.0, help="Tolerance window for temporal precision (seconds)."),
    run_id: str | None = typer.Option(None, help="Custom run ID. Auto-generated if not provided."),
) -> None:
    """Run an end-to-end eval campaign for a given dataset and model."""
    if dataset not in _DATASET_LOADERS:
        typer.echo(f"Unknown dataset '{dataset}'. Choose from: {list(_DATASET_LOADERS)}", err=True)
        raise typer.Exit(code=1)

    effective_run_id = run_id or f"{dataset}_{_today()}_{uuid.uuid4().hex[:6]}"
    run_output_dir = output / effective_run_id

    console.print(f"[bold]SENTINEL[/bold] — run [cyan]{effective_run_id}[/cyan]")
    console.print(f"  Dataset : {dataset}")
    console.print(f"  Model   : {model}")
    console.print(f"  Output  : {run_output_dir}")
    console.rule()

    # Dynamically import the loader to avoid importing optional heavy deps at startup.
    loader = _load_dataset_class(_DATASET_LOADERS[dataset])()

    anomaly_agent = AnomalyAgent(model=model)
    scoring_agent = EvalScoringAgent(temporal_tolerance_seconds=temporal_tolerance)

    from sentinel.fusion import signal_packager

    detections = []
    labels = []
    n = 0

    for window, label in loader.iter_segments(subject=subject):
        prompt_context = signal_packager.package(window)
        detection = anomaly_agent.detect(prompt_context)
        detections.append(detection)
        labels.append(label)
        n += 1
        if n % 50 == 0:
            console.print(f"  Processed {n} segments…")

    console.print(f"  Total segments evaluated: {n}")
    scores = scoring_agent.score_batch(detections, labels)

    report = report_generator.build_report(
        run_id=effective_run_id,
        model=model,
        dataset=dataset,
        scores=scores,
        temporal_tolerance_seconds=temporal_tolerance,
    )

    out_path = report_generator.save_report(report, run_output_dir)
    console.print(f"\n[green]Report written to {out_path}[/green]")
    console.rule()
    console.print(report_generator.format_summary(report))


def _load_dataset_class(dotted_path: str):  # type: ignore[return]
    """Import and return a class from a dotted module path."""
    module_path, class_name = dotted_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _today() -> str:
    from datetime import date
    return date.today().strftime("%Y%m%d")


if __name__ == "__main__":
    app()
