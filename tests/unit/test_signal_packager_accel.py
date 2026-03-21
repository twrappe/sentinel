"""Unit tests for signal_packager — accelerometer modality.

Focuses exclusively on how ``package`` handles accelerometer channels in a
``BiosignalWindow``. The default fixture here provides only accel channels
(EEG and EMG are empty) so each test is unambiguously exercising accelerometer
behaviour. Tests assert structural correctness of the text output without
coupling to exact formatting details.
"""

import numpy as np

from sentinel.fusion.signal_packager import BiosignalWindow, package


def _make_window(**kwargs):
    """Return a ``BiosignalWindow`` with accelerometer channels by default.

    EEG and EMG are omitted so the output contains only the Accelerometer
    section, making assertions unambiguous. Override any field via kwargs.
    """
    defaults = dict(
        duration_seconds=10.0,
        sample_rate_hz=100.0,  # Accel is typically sampled at lower rates than EEG/EMG
        eeg_channels={},
        emg_channels={},
        accel_channels={"x": np.zeros(1000), "y": np.zeros(1000), "z": np.ones(1000)},
    )
    defaults.update(kwargs)
    return BiosignalWindow(**defaults)


class TestSignalPackagerAccel:
    """Verifies ``package`` correctly renders the Accelerometer section of the prompt.

    Each test targets one aspect of accelerometer channel handling: section
    presence, axis name inclusion, multi-axis coverage, and graceful omission
    when no accel data is present. Three-axis (x/y/z) fixtures are used as the
    canonical case since that is the standard configuration for wrist-worn devices.
    """

    def test_accel_section_present_when_channels_provided(self):
        # A window with accel channels must produce output containing an
        # Accelerometer section header so the model can identify the modality.
        result = package(_make_window())
        assert "Accelerometer" in result

    def test_accel_section_absent_when_no_channels(self):
        # When accel_channels is empty, the Accelerometer section must be omitted
        # entirely — an empty heading with no channel data would mislead the model.
        result = package(_make_window(accel_channels={}))
        assert "Accelerometer" not in result

    def test_accel_axis_names_appear_in_output(self):
        # Each axis name must be present in the output so the model can cite
        # specific axes when describing motion artefacts or events.
        result = package(_make_window())
        assert "x" in result
        assert "y" in result
        assert "z" in result

    def test_accel_single_axis(self):
        # A window with only one axis must still produce an Accelerometer section
        # and include that axis name — section rendering must not require all
        # three axes to be present.
        result = package(_make_window(accel_channels={"z": np.zeros(1000)}))
        assert "Accelerometer" in result
        assert "z" in result

    def test_accel_channel_summary_values_appear(self):
        # The statistical summary for an all-ones axis must surface a non-zero
        # value, confirming that axis amplitudes are reflected in the output
        # rather than silently zeroed.
        result = package(_make_window(accel_channels={"z": np.ones(1000)}))
        # rms and peak of an all-ones signal are both 1.0
        assert "1.0" in result or "1.00" in result

    def test_eeg_section_absent_in_accel_only_window(self):
        # When only accel channels are provided, the EEG section must not appear —
        # absent modalities should produce no section heading.
        result = package(_make_window())
        assert "EEG" not in result

    def test_emg_section_absent_in_accel_only_window(self):
        # When only accel channels are provided, the EMG section must not appear —
        # absent modalities should produce no section heading.
        result = package(_make_window())
        assert "EMG" not in result
