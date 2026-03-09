"""Tests for certificate generation, serialisation, and verification."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from cymatic_seal.seal.certificate import (
    SealCertificate,
    generate_certificate,
    hash_file,
    _hash_audio,
)
from cymatic_seal.verify.verifier import verify_seal, VerificationResult
from cymatic_seal.audio import save_audio


def _dummy_artefacts(n_channels=2, n_samples=44100):
    """Create plausible dummy sealing artefacts for testing."""
    original = np.random.randn(n_channels, n_samples).astype(np.float32) * 0.5
    perturbation = np.random.randn(n_channels, n_samples).astype(np.float32) * 0.001
    sealed = np.clip(original + perturbation, -1, 1).astype(np.float32)
    n_bins = 221
    n_frames = 20
    freqs = np.linspace(0, 22050, n_bins, dtype=np.float32)
    threshold = np.random.rand(n_frames, n_bins).astype(np.float32) * 0.01
    return original, sealed, perturbation, freqs, threshold


class TestCertificate:
    def test_json_round_trip(self):
        orig, sealed, pert, freqs, thr = _dummy_artefacts()
        cert = generate_certificate(orig, sealed, pert, freqs, thr, sr=44100)

        text = cert.to_json()
        loaded = SealCertificate.from_json(text)

        assert loaded.version == cert.version
        assert loaded.sealed_audio_hash_sha256 == cert.sealed_audio_hash_sha256
        assert loaded.algorithm == cert.algorithm

    def test_hash_determinism(self):
        wav = np.random.randn(2, 1000).astype(np.float32)
        h1 = _hash_audio(wav)
        h2 = _hash_audio(wav)
        assert h1 == h2

    def test_different_audio_different_hash(self):
        a = np.random.randn(2, 1000).astype(np.float32)
        b = np.random.randn(2, 1000).astype(np.float32)
        assert _hash_audio(a) != _hash_audio(b)


class TestVerification:
    def test_matching_audio_verifies(self, tmp_path: Path):
        """Certificate built with file-path hashing must verify."""
        orig, sealed, pert, freqs, thr = _dummy_artefacts()

        orig_path = tmp_path / "original.wav"
        save_audio(str(orig_path), orig, sr=44100)

        audio_path = tmp_path / "sealed.wav"
        save_audio(str(audio_path), sealed, sr=44100)

        cert = generate_certificate(
            orig, sealed, pert, freqs, thr, sr=44100,
            original_file_path=str(orig_path),
            sealed_file_path=str(audio_path),
        )

        cert_path = tmp_path / "cert.json"
        cert_path.write_text(cert.to_json(), encoding="utf-8")

        result = verify_seal(audio_path, cert_path)
        assert result.verified is True

    def test_tampered_audio_fails(self, tmp_path: Path):
        orig, sealed, pert, freqs, thr = _dummy_artefacts()

        orig_path = tmp_path / "original.wav"
        save_audio(str(orig_path), orig, sr=44100)

        sealed_path = tmp_path / "sealed.wav"
        save_audio(str(sealed_path), sealed, sr=44100)

        cert = generate_certificate(
            orig, sealed, pert, freqs, thr, sr=44100,
            original_file_path=str(orig_path),
            sealed_file_path=str(sealed_path),
        )

        # Write a different (tampered) file under a new name
        tampered = sealed + 0.01
        tampered_path = tmp_path / "tampered.wav"
        save_audio(str(tampered_path), tampered, sr=44100)

        cert_path = tmp_path / "cert.json"
        cert_path.write_text(cert.to_json(), encoding="utf-8")

        result = verify_seal(tampered_path, cert_path)
        assert result.verified is False
