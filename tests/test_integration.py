"""Integration test: full seal → verify pipeline.

This test requires the ``demucs`` package to be installed and will use
whatever device is available (CPU or CUDA).  It is intentionally slow
and should be run separately from the fast unit-test suite.

    pytest tests/test_integration.py -v -s
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

# Skip the entire module if demucs is not installed.
pytest.importorskip("demucs", reason="demucs not installed — skipping integration tests")

from cymatic_seal.audio import save_audio
from cymatic_seal.seal import seal_audio
from cymatic_seal.verify import verify_seal


def _make_test_track(path: Path, sr: int = 44100, dur: float = 3.0):
    """Write a short stereo sine-sweep to *path*."""
    t = np.linspace(0, dur, int(sr * dur), endpoint=False, dtype=np.float32)
    freq = np.linspace(200, 4000, len(t))
    mono = 0.6 * np.sin(2 * np.pi * freq * t / sr * np.cumsum(np.ones_like(t)))
    stereo = np.stack([mono, mono * 0.9])
    save_audio(str(path), stereo, sr=sr)
    return stereo


class TestFullPipeline:
    def test_seal_and_verify(self, tmp_path: Path):
        input_path = tmp_path / "track.wav"
        output_path = tmp_path / "track_sealed.wav"
        cert_path = tmp_path / "cert.json"

        original = _make_test_track(input_path)

        sealed, cert = seal_audio(
            input_path,
            output_path=output_path,
            certificate_path=cert_path,
            artist="Test Artist",
            title="Test Track",
            method="fgsm",
            steps=1,
            epsilon=0.02,
            device="cpu",
        )

        # Sealed audio should be close to original (imperceptible)
        max_diff = np.max(np.abs(sealed - original))
        assert max_diff <= 0.025, f"Max perturbation {max_diff} exceeds epsilon."

        # Certificate should exist and contain expected fields
        assert cert.sealed_audio_hash_sha256
        assert cert.artist == "Test Artist"
        assert cert.duration_seconds > 0

        # Verification should pass
        result = verify_seal(output_path, cert_path)
        assert result.verified, f"Verification failed: {result.reason}"

    def test_seal_modifies_audio(self, tmp_path: Path):
        """The sealed audio must differ from the original."""
        input_path = tmp_path / "track.wav"
        output_path = tmp_path / "track_sealed.wav"

        _make_test_track(input_path)

        sealed, _ = seal_audio(
            input_path,
            output_path=output_path,
            method="fgsm",
            steps=1,
            device="cpu",
        )

        from cymatic_seal.audio import load_audio
        original, _ = load_audio(str(input_path))

        assert not np.allclose(sealed, original, atol=1e-8), (
            "Sealed audio is identical to original — perturbation was not applied."
        )
