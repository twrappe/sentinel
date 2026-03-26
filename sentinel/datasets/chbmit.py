"""CHB-MIT Scalp EEG dataset loader.

Dataset: https://physionet.org/content/chbmit/1.0.0/
Signals: EEG (scalp, 23 channels at 256 Hz)
Events:  Seizure onset / offset annotations

Quickstart::

    # Download chb01 (~1-2 GB) and preprocess into windows
    python -m sentinel.datasets.chbmit --download --preprocess --subject chb01

    # Verify
    python -m sentinel.datasets.chbmit --summary --subject chb01
"""

from __future__ import annotations

import argparse
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import mne
import numpy as np

from sentinel.agents.eval_scoring_agent import GroundTruthLabel
from sentinel.fusion.signal_packager import BiosignalWindow

mne.set_log_level("WARNING")

# Default root for raw data.  Override via SENTINEL_DATA_DIR env var or --data-dir flag.
_DEFAULT_DATA_DIR = Path("data/raw/chbmit")
_DEFAULT_PROCESSED_DIR = Path("data/processed/chbmit")
_DEFAULT_GROUND_TRUTH_DIR = Path("data/ground_truth/chbmit")

_PHYSIONET_BASE = "https://physionet.org/files/chbmit/1.0.0"

# Minimum overlap fraction for a window to be labelled as a seizure.
_DEFAULT_SEIZURE_OVERLAP_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SeizureInterval:
    """A single annotated seizure interval within one EDF recording."""
    onset_seconds: float
    offset_seconds: float

    @property
    def duration_seconds(self) -> float:
        return self.offset_seconds - self.onset_seconds


# Mapping: edf_filename → list of seizure intervals for that file.
SubjectAnnotations = dict[str, list[SeizureInterval]]


# ---------------------------------------------------------------------------
# 1.1  Download
# ---------------------------------------------------------------------------

def _download(data_dir: Path, subject: str = "chb01") -> None:
    """Download EDF files and the summary text for one CHB-MIT subject.

    Fetches the directory index from PhysioNet and downloads all ``.edf``
    files and the ``*-summary.txt`` file for the given subject.  Files that
    already exist on disk are skipped so re-runs are idempotent.

    Args:
        data_dir: Root directory for raw data (e.g. ``data/raw/chbmit``).
        subject:  Subject folder name on PhysioNet (e.g. ``"chb01"``).
    """
    subject_dir = data_dir / subject
    subject_dir.mkdir(parents=True, exist_ok=True)

    base_url = f"{_PHYSIONET_BASE}/{subject}"

    # Fetch the directory listing to discover file names.
    index_url = f"{base_url}/"
    print(f"Fetching directory index: {index_url}")
    with urllib.request.urlopen(index_url) as resp:
        html = resp.read().decode()

    # Extract .edf and -summary.txt hrefs from the PhysioNet directory listing.
    filenames = re.findall(r'href="([^"]+\.(?:edf|txt))"', html, re.IGNORECASE)
    # Keep only files directly in this subject folder (no subdirectory paths).
    filenames = [f for f in filenames if "/" not in f]

    if not filenames:
        raise RuntimeError(
            f"No .edf or .txt files found at {index_url}. "
            "Check the URL or your internet connection."
        )

    print(f"Found {len(filenames)} file(s) to download for {subject}.")
    for filename in filenames:
        dest = subject_dir / filename
        if dest.exists():
            print(f"  [skip] {filename}")
            continue
        url = f"{base_url}/{filename}"
        print(f"  [down] {filename}")
        urllib.request.urlretrieve(url, dest)

    print(f"Download complete → {subject_dir}")


# ---------------------------------------------------------------------------
# 1.2  Summary file parser
# ---------------------------------------------------------------------------

def _parse_summary(subject_dir: Path, subject: str) -> SubjectAnnotations:
    """Parse the CHB-MIT ``*-summary.txt`` file into structured annotations.

    The summary file uses a plaintext format where each EDF file is described
    with a block containing (among other things) the seizure start and end
    times in elapsed seconds from the recording start.

    CHB-MIT has minor format inconsistencies across subjects:
    - ``"Seizure Start Time: N seconds"`` (most subjects)
    - ``"Seizure 1 Start Time: N seconds"`` (some subjects, multiple seizures)

    Both variants are handled.

    Args:
        subject_dir: Path to the subject's raw data directory.
        subject:     Subject name used to find the summary file (e.g. "chb01").

    Returns:
        ``SubjectAnnotations``: dict mapping EDF filename → list of
        ``SeizureInterval`` objects.
    """
    summary_candidates = list(subject_dir.glob("*-summary.txt"))
    if not summary_candidates:
        raise FileNotFoundError(
            f"No summary file found in {subject_dir}. "
            "Run --download first."
        )
    summary_path = summary_candidates[0]
    text = summary_path.read_text(encoding="utf-8", errors="replace")

    annotations: SubjectAnnotations = {}
    current_file: str | None = None
    pending_onset: float | None = None

    # Regex patterns — written to handle both "Seizure Start Time" and
    # "Seizure N Start Time" variants.
    file_re = re.compile(r"File Name:\s*(\S+\.edf)", re.IGNORECASE)
    onset_re = re.compile(r"Seizure(?:\s+\d+)?\s+Start Time:\s*(\d+)\s*second", re.IGNORECASE)
    offset_re = re.compile(r"Seizure(?:\s+\d+)?\s+End Time:\s*(\d+)\s*second", re.IGNORECASE)

    for line in text.splitlines():
        file_match = file_re.search(line)
        if file_match:
            current_file = file_match.group(1).strip()
            if current_file not in annotations:
                annotations[current_file] = []
            pending_onset = None
            continue

        onset_match = onset_re.search(line)
        if onset_match and current_file is not None:
            pending_onset = float(onset_match.group(1))
            continue

        offset_match = offset_re.search(line)
        if offset_match and current_file is not None and pending_onset is not None:
            offset = float(offset_match.group(1))
            annotations[current_file].append(
                SeizureInterval(onset_seconds=pending_onset, offset_seconds=offset)
            )
            pending_onset = None

    return annotations


# ---------------------------------------------------------------------------
# 1.3  EDF reader + windowing
# ---------------------------------------------------------------------------

def _label_window(
    win_start: float,
    win_end: float,
    seizure_intervals: list[SeizureInterval],
    overlap_threshold: float,
) -> bool:
    """Return True if this window overlaps sufficiently with any seizure interval.

    A window is labelled as a seizure if the fraction of the window duration
    covered by any single annotated seizure interval meets or exceeds
    ``overlap_threshold``.
    """
    win_duration = win_end - win_start
    for interval in seizure_intervals:
        overlap = max(0.0, min(win_end, interval.offset_seconds) - max(win_start, interval.onset_seconds))
        if overlap / win_duration >= overlap_threshold:
            return True
    return False


def _window_recording(
    edf_path: Path,
    seizure_intervals: list[SeizureInterval],
    window_seconds: float = 10.0,
    stride_seconds: float | None = None,
    overlap_threshold: float = _DEFAULT_SEIZURE_OVERLAP_THRESHOLD,
) -> list[tuple[BiosignalWindow, GroundTruthLabel]]:
    """Read one EDF file and slice it into labelled ``BiosignalWindow`` objects.

    Args:
        edf_path:          Path to the ``.edf`` file.
        seizure_intervals: Annotated seizure intervals for this recording.
        window_seconds:    Duration of each window in seconds (default 10s).
        stride_seconds:    Step between window starts.  Defaults to
                           ``window_seconds`` (no overlap).
        overlap_threshold: Minimum fraction of window duration that must
                           overlap with a seizure annotation for the window
                           to be labelled as a seizure.

    Returns:
        List of ``(BiosignalWindow, GroundTruthLabel)`` pairs.
    """
    if stride_seconds is None:
        stride_seconds = window_seconds

    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    sfreq = raw.info["sfreq"]
    n_samples_window = int(window_seconds * sfreq)
    n_samples_stride = int(stride_seconds * sfreq)
    total_samples = raw.n_times

    # Normalise channel names: strip whitespace, upper-case.
    ch_names = [ch.strip().upper() for ch in raw.ch_names]

    data, _ = raw.get_data(return_times=True)  # shape: (n_channels, n_samples)

    segments: list[tuple[BiosignalWindow, GroundTruthLabel]] = []
    win_start_sample = 0

    while win_start_sample + n_samples_window <= total_samples:
        win_end_sample = win_start_sample + n_samples_window
        win_start_sec = win_start_sample / sfreq
        win_end_sec = win_end_sample / sfreq

        window_data = data[:, win_start_sample:win_end_sample]

        eeg_channels = {
            ch_names[i]: window_data[i] for i in range(len(ch_names))
        }

        has_seizure = _label_window(
            win_start_sec, win_end_sec, seizure_intervals, overlap_threshold
        )

        # Find the earliest seizure onset within this window (if any) for
        # the GroundTruthLabel onset field.
        onset_in_window: float | None = None
        offset_in_window: float | None = None
        if has_seizure:
            for interval in seizure_intervals:
                if interval.onset_seconds < win_end_sec and interval.offset_seconds > win_start_sec:
                    local_onset = max(0.0, interval.onset_seconds - win_start_sec)
                    local_offset = min(window_seconds, interval.offset_seconds - win_start_sec)
                    if onset_in_window is None or local_onset < onset_in_window:
                        onset_in_window = local_onset
                        offset_in_window = local_offset

        bw = BiosignalWindow(
            duration_seconds=window_seconds,
            sample_rate_hz=sfreq,
            eeg_channels=eeg_channels,
            metadata={
                "source_file": edf_path.name,
                "window_start_seconds": win_start_sec,
            },
        )

        label = GroundTruthLabel(
            has_event=has_seizure,
            event_type="seizure" if has_seizure else None,
            onset_seconds=onset_in_window,
            channels=list(eeg_channels.keys()),
        )

        segments.append((bw, label))
        win_start_sample += n_samples_stride

    return segments


# ---------------------------------------------------------------------------
# 1.4  CHBMITLoader
# ---------------------------------------------------------------------------

class CHBMITLoader:
    """Loads windowed EEG segments and seizure annotations from CHB-MIT.

    Usage::

        loader = CHBMITLoader(raw_dir=Path("data/raw/chbmit"))
        for window, label in loader.iter_segments(subject="chb01"):
            ...
    """

    def __init__(
        self,
        raw_dir: Path = _DEFAULT_DATA_DIR,
        window_seconds: float = 10.0,
        stride_seconds: float | None = None,
        seizure_overlap_threshold: float = _DEFAULT_SEIZURE_OVERLAP_THRESHOLD,
    ) -> None:
        self.raw_dir = raw_dir
        self.window_seconds = window_seconds
        self.stride_seconds = stride_seconds
        self.seizure_overlap_threshold = seizure_overlap_threshold

    def iter_segments(
        self, subject: str | None = None
    ) -> Iterator[tuple[BiosignalWindow, GroundTruthLabel]]:
        """Yield ``(BiosignalWindow, GroundTruthLabel)`` pairs.

        Args:
            subject: Restrict to a single subject folder (e.g. ``"chb01"``).
                     If None, iterates all subject subdirectories found under
                     ``raw_dir``.

        Raises:
            FileNotFoundError: If the raw data directory or subject folder
                               does not exist.  Run ``--download`` first.
        """
        if not self.raw_dir.exists():
            raise FileNotFoundError(
                f"Raw data directory not found: {self.raw_dir}\n"
                "Run: python -m sentinel.datasets.chbmit --download"
            )

        subject_dirs = (
            [self.raw_dir / subject]
            if subject
            else sorted(p for p in self.raw_dir.iterdir() if p.is_dir())
        )

        for subject_dir in subject_dirs:
            if not subject_dir.exists():
                raise FileNotFoundError(
                    f"Subject directory not found: {subject_dir}\n"
                    "Run: python -m sentinel.datasets.chbmit --download "
                    f"--subject {subject_dir.name}"
                )

            annotations = _parse_summary(subject_dir, subject_dir.name)

            for edf_path in sorted(subject_dir.glob("*.edf")):
                seizure_intervals = annotations.get(edf_path.name, [])
                segments = _window_recording(
                    edf_path,
                    seizure_intervals,
                    window_seconds=self.window_seconds,
                    stride_seconds=self.stride_seconds,
                    overlap_threshold=self.seizure_overlap_threshold,
                )
                yield from segments


# ---------------------------------------------------------------------------
# 1.5  Preprocessing pipeline + CLI
# ---------------------------------------------------------------------------

def _summary(raw_dir: Path, subject: str) -> None:
    """Print a human-readable annotation summary without running windowing.

    Parses the subject's summary file and prints the number of EDF recordings,
    total annotated seizures, and per-file seizure intervals.
    """
    subject_dir = raw_dir / subject
    annotations = _parse_summary(subject_dir, subject)

    total_seizures = sum(len(v) for v in annotations.values())
    n_files = len(annotations)
    print(f"{subject}: {n_files} recording(s), {total_seizures} annotated seizure(s)")
    print()
    for filename, intervals in sorted(annotations.items()):
        if intervals:
            for i, iv in enumerate(intervals, 1):
                print(f"  {filename}  seizure {i}: {iv.onset_seconds}s – {iv.offset_seconds}s ({iv.duration_seconds:.1f}s)")
        else:
            print(f"  {filename}  (no seizures)")


def _preprocess(raw_dir: Path, subject: str) -> None:
    """Run the full pipeline for one subject and print a segment count summary.

    This is the Definition of Done check: produces the line
    ``chb01: N segments (M seizure, K non-seizure)``.
    """
    loader = CHBMITLoader(raw_dir=raw_dir)
    n_seizure = 0
    n_non_seizure = 0

    for _, label in loader.iter_segments(subject=subject):
        if label.has_event:
            n_seizure += 1
        else:
            n_non_seizure += 1

    total = n_seizure + n_non_seizure
    print(f"{subject}: {total} segments ({n_seizure} seizure, {n_non_seizure} non-seizure)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CHB-MIT dataset utilities")
    parser.add_argument("--download", action="store_true", help="Download raw dataset for subject")
    parser.add_argument("--preprocess", action="store_true", help="Run pipeline and print segment summary")
    parser.add_argument("--summary", action="store_true", help="Print annotation summary (no windowing)")
    parser.add_argument("--subject", type=str, default="chb01", help="Subject ID (default: chb01)")
    parser.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA_DIR, dest="data_dir")
    args = parser.parse_args()

    if args.download:
        _download(args.data_dir, subject=args.subject)
    if args.preprocess:
        _preprocess(args.data_dir, subject=args.subject)
    if args.summary:
        _summary(args.data_dir, subject=args.subject)
    if not args.download and not args.preprocess and not args.summary:
        parser.print_help()
