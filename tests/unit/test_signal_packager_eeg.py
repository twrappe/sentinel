"""Unit tests for signal_packager.

``signal_packager.package`` converts a ``BiosignalWindow`` into the text prompt
that is passed to the LLM. These tests verify structural correctness of the
output — that the right sections appear (or are omitted) and that channel names
and metadata are surfaced — without asserting on the exact wording, which is
subject to iteration as the prompt design matures.
"""

import numpy as np
import pytest

from sentinel.fusion.signal_packager import BiosignalWindow, package


def _make_window(**kwargs):
    defaults = dict(
        duration_seconds=10.0,
        sample_rate_hz=256.0,
        eeg_channels={"F3": np.zeros(2560), "C3": np.ones(2560)},
        emg_channels={},
        accel_channels={},
    )
    defaults.update(kwargs)
    return BiosignalWindow(**defaults)


class TestSignalPackager:
    """Verifies that signal_packager.package produces well-structured LLM prompt text.

    Tests focus on the presence or absence of named sections and identifiers
    in the output string. They do not assert exact formatting so that the
    prompt wording can evolve without breaking the suite.
    """

    def test_package_returns_string(self):
        # The most basic contract: package must always return a plain string
        # that can be passed directly as LLM message content.
        result = package(_make_window())
        assert isinstance(result, str)

    def test_package_includes_eeg_section(self):
        # A window with EEG channels must produce output that contains an
        # EEG section header so the model knows which modality it is reading.
        result = package(_make_window())
        assert "EEG" in result

    def test_package_includes_emg_section_when_present(self):
        # When EMG channels are provided, the output must contain an EMG section.
        window = _make_window(emg_channels={"left": np.zeros(2560)})
        result = package(window)
        assert "EMG" in result

    def test_package_includes_accel_section_when_present(self):
        # When accelerometer channels are provided, the output must contain
        # an Accelerometer section.
        window = _make_window(accel_channels={"x": np.zeros(2560)})
        result = package(window)
        assert "Accelerometer" in result

    def test_package_includes_channel_names(self):
        # Individual channel names must appear in the output so the model can
        # cite specific channels in its evidence_channels field.
        result = package(_make_window())
        assert "F3" in result
        assert "C3" in result

    def test_package_omits_emg_section_when_absent(self):
        # When no EMG channels are supplied, the EMG section must be omitted
        # entirely — an empty section heading would be misleading to the model.
        result = package(_make_window(emg_channels={}))
        assert "EMG" not in result

    def test_package_includes_metadata(self):
        # Metadata values (e.g. subject ID, segment index) must be present in
        # the output so the model has context about the segment's provenance.
        window = _make_window(metadata={"subject": "chb01", "segment": 5})
        result = package(window)
        assert "chb01" in result
