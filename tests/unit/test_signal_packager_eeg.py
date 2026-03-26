"""Unit tests for signal_packager — EEG modality.

Focuses exclusively on how ``package`` handles EEG channels in a
``BiosignalWindow``. The default fixture here provides only EEG channels
(EMG and accel are empty) so each test is unambiguously exercising EEG
behaviour. Tests assert structural correctness of the text output without
coupling to exact formatting details.
"""

import numpy as np

from sentinel.fusion.signal_packager import BiosignalWindow, package


def _make_window(**kwargs):
    """Return a ``BiosignalWindow`` with EEG channels by default.

    EMG and accel are omitted so the output contains only the EEG section,
    making assertions unambiguous. Override any field via kwargs.
    """
    defaults = dict(
        duration_seconds=10.0,
        sample_rate_hz=256.0,  # Standard scalp EEG sample rate
        eeg_channels={"F3": np.zeros(2560), "C3": np.ones(2560)},
        emg_channels={},
        accel_channels={},
    )
    defaults.update(kwargs)
    return BiosignalWindow(**defaults)


class TestSignalPackager:
    """Verifies ``package`` correctly renders the EEG section of the prompt.

    Each test targets one aspect of EEG channel handling: section presence,
    channel name inclusion, multi-channel coverage, graceful omission when no
    EEG data is present, and metadata surfacing.
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

    def test_eeg_section_absent_when_no_channels(self):
        # When eeg_channels is empty, the EEG section must be omitted entirely —
        # an empty heading with no channel data would mislead the model.
        result = package(_make_window(eeg_channels={}))
        assert "EEG" not in result

    def test_package_includes_channel_names(self):
        # Individual channel names must appear in the output so the model can
        # cite specific electrodes in its evidence_channels field.
        result = package(_make_window())
        assert "F3" in result
        assert "C3" in result

    def test_eeg_channel_summary_values_appear(self):
        # The statistical summary for an all-ones channel must surface a
        # non-zero value, confirming that channel amplitudes are reflected
        # in the output rather than silently zeroed.
        result = package(_make_window(eeg_channels={"C3": np.ones(2560)}))
        # rms and peak of an all-ones signal are both 1.0
        assert "1.0" in result or "1.00" in result

    def test_emg_section_absent_in_eeg_only_window(self):
        # When only EEG channels are provided, the EMG section must not appear —
        # absent modalities should produce no section heading.
        result = package(_make_window())
        assert "EMG" not in result

    def test_accel_section_absent_in_eeg_only_window(self):
        # When only EEG channels are provided, the Accelerometer section must
        # not appear — absent modalities should produce no section heading.
        result = package(_make_window())
        assert "Accelerometer" not in result

    def test_package_includes_metadata(self):
        # Metadata values (e.g. subject ID, segment index) must be present in
        # the output so the model has context about the segment's provenance.
        window = _make_window(metadata={"subject": "chb01", "segment": 5})
        result = package(window)
        assert "chb01" in result
