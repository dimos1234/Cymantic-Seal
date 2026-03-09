"""Tests for the psychoacoustic masking model."""

import numpy as np
import pytest

from cymatic_seal.psychoacoustics.masking import (
    compute_masking_threshold,
    _hz_to_bark,
    _absolute_threshold_of_hearing,
)


def _sine_mono(freq: float = 1000.0, sr: int = 44100, dur: float = 0.2):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False, dtype=np.float32)
    return 0.8 * np.sin(2 * np.pi * freq * t)


class TestHelpers:
    def test_bark_monotonic(self):
        freqs = np.linspace(20, 20000, 500)
        bark = _hz_to_bark(freqs)
        assert np.all(np.diff(bark) > 0), "Bark scale should be monotonically increasing."

    def test_ath_minimum_near_3khz(self):
        freqs = np.linspace(100, 15000, 1000)
        ath = _absolute_threshold_of_hearing(freqs)
        min_idx = np.argmin(ath)
        min_freq = freqs[min_idx]
        assert 2000 < min_freq < 5000, (
            f"ATH minimum at {min_freq} Hz — expected near 3–4 kHz."
        )


class TestMaskingThreshold:
    def test_output_shape(self):
        wav = _sine_mono(dur=0.1)
        thr, freqs = compute_masking_threshold(wav, sr=44100)

        assert thr.ndim == 2
        assert thr.shape[1] == len(freqs)
        assert thr.shape[0] >= 1

    def test_positive_thresholds(self):
        wav = _sine_mono()
        thr, _ = compute_masking_threshold(wav, sr=44100)
        assert np.all(thr > 0), "All thresholds must be positive (linear amplitude)."

    def test_louder_signal_raises_threshold(self):
        """A louder masker should produce a higher masking threshold overall."""
        quiet = _sine_mono(freq=1000) * 0.1
        loud = _sine_mono(freq=1000) * 0.9

        thr_q, _ = compute_masking_threshold(quiet, sr=44100)
        thr_l, _ = compute_masking_threshold(loud, sr=44100)

        assert thr_l.mean() > thr_q.mean()

    def test_temporal_masking_raises_threshold(self):
        """With temporal masking enabled, thresholds around transients should
        be at least as high as without."""
        wav = _sine_mono(dur=0.3)
        # Insert a loud click at the midpoint
        mid = len(wav) // 2
        wav[mid : mid + 50] = 0.95

        thr_no, _ = compute_masking_threshold(wav, sr=44100, temporal=False)
        thr_yes, _ = compute_masking_threshold(wav, sr=44100, temporal=True)

        assert thr_yes.mean() >= thr_no.mean()
