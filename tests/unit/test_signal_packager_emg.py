"""Unit tests for signal_packager — EMG modality.

Focuses exclusively on how ``package`` handles EMG channels in a
``BiosignalWindow``. The default fixture here provides only EMG channels
(EEG and accel are empty) so each test is unambiguously exercising EMG
behaviour. Tests assert structural correctness of the text output without
coupling to exact formatting details.
"""

import numpy as np

from sentinel.fusion.signal_packager import BiosignalWindow, package


def _make_window(**kwargs):
    """Return a ``BiosignalWindow`` with EMG channels by default.

    EEG and accel are omitted so the output contains only the EMG section,
    making assertions unambiguous. Override any field via kwargs.
    """
    defaults = dict(
        duration_seconds=10.0,
        sample_rate_hz=2000.0,  # EMG is typically sampled at higher rates than EEG
        eeg_channels={},
        emg_channels={"left_bicep": np.zeros(20000), "right_bicep": np.ones(20000)},
        accel_channels={},
    )
    defaults.update(kwargs)
    return BiosignalWindow(**defaults)


class TestSignalPackagerEMG:
    """Verifies ``package`` correctly renders the EMG section of the prompt.

    Each test targets one aspect of EMG channel handling: section presence,
    channel name inclusion, multi-channel coverage, and graceful omission
    when no EMG data is present.
    """

    def test_emg_section_present_when_channels_provided(self):
        # A window with EMG channels must produce output containing an EMG
        # section header so the model can identify the modality.
        result = package(_make_window())
        assert "EMG" in result

    def test_emg_section_absent_when_no_channels(self):
        # When emg_channels is empty, the EMG section must be omitted entirely —
        # an empty heading with no channel data would mislead the model.
        result = package(_make_window(emg_channels={}))
        assert "EMG" not in result

    def test_emg_channel_names_appear_in_output(self):
        # Each EMG channel name must be present in the output so the model can
        # cite specific muscles in its evidence_channels field.
        result = package(_make_window())
        assert "left_bicep" in result
        assert "right_bicep" in result

    def test_emg_single_channel(self):
        # A window with a single EMG channel must still produce an EMG section
        # and include that channel's name — section rendering must not require
        # multiple channels to activate.
        result = package(_make_window(emg_channels={"forearm": np.zeros(20000)}))
        assert "EMG" in result
        assert "forearm" in result

    def test_emg_channel_summary_values_appear(self):
        # The statistical summary for an all-ones channel must surface a
        # non-zero value, confirming that channel amplitudes are reflected
        # in the output rather than silently zeroed.
        result = package(_make_window(emg_channels={"active": np.ones(20000)}))
        # rms and peak of an all-ones signal are both 1.0
        assert "1.0" in result or "1.00" in result

    def test_eeg_section_absent_in_emg_only_window(self):
        # When only EMG channels are provided, the EEG section must not appear —
        # absent modalities should produce no section heading.
        result = package(_make_window())
        assert "EEG" not in result

    def test_accel_section_absent_in_emg_only_window(self):
        # When only EMG channels are provided, the Accelerometer section must
        # not appear — absent modalities should produce no section heading.
        result = package(_make_window())
        assert "Accelerometer" not in result
