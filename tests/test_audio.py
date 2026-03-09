"""Tests for the audio I/O module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from cymatic_seal.audio import load_audio, save_audio, normalize, resample


def _sine(freq: float = 440.0, sr: int = 44100, dur: float = 0.5, channels: int = 2):
    """Generate a stereo sine wave."""
    t = np.linspace(0, dur, int(sr * dur), endpoint=False, dtype=np.float32)
    mono = 0.5 * np.sin(2 * np.pi * freq * t)
    return np.stack([mono] * channels)


class TestSaveAndLoad:
    def test_wav_round_trip(self, tmp_path: Path):
        original = _sine()
        path = tmp_path / "test.wav"
        save_audio(str(path), original, sr=44100)
        loaded, sr = load_audio(str(path), sr=44100)

        assert sr == 44100
        assert loaded.shape == original.shape
        np.testing.assert_allclose(loaded, original, atol=1e-4)

    def test_flac_round_trip(self, tmp_path: Path):
        original = _sine()
        path = tmp_path / "test.flac"
        save_audio(str(path), original, sr=44100)
        loaded, sr = load_audio(str(path), sr=44100)

        assert sr == 44100
        assert loaded.shape[0] == original.shape[0]
        # FLAC is lossless but quantised to 16-bit by default
        np.testing.assert_allclose(loaded, original, atol=0.01)

    def test_mono_loading(self, tmp_path: Path):
        original = _sine(channels=2)
        path = tmp_path / "stereo.wav"
        save_audio(str(path), original)
        loaded, _ = load_audio(str(path), mono=True)

        assert loaded.shape[0] == 1


class TestResample:
    def test_identity(self):
        wav = _sine(sr=44100)
        out = resample(wav, 44100, 44100)
        np.testing.assert_array_equal(wav, out)

    def test_downsample(self):
        wav = _sine(sr=44100, dur=1.0)
        out = resample(wav, 44100, 22050)
        expected_len = wav.shape[1] // 2
        assert abs(out.shape[1] - expected_len) <= 2


class TestNormalize:
    def test_peak(self):
        wav = _sine() * 0.1
        normed = normalize(wav, peak=0.9)
        assert abs(np.max(np.abs(normed)) - 0.9) < 1e-5

    def test_silence(self):
        wav = np.zeros((2, 1000), dtype=np.float32)
        normed = normalize(wav)
        np.testing.assert_array_equal(normed, wav)
