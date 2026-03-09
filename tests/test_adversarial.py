"""Tests for the adversarial engine (unit-level, without requiring Demucs).

Full integration tests that actually load Demucs are in test_integration.py
and require a GPU-capable environment with demucs installed.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed — skipping adversarial unit tests")
from cymatic_seal.adversarial.engine import AdversarialEngine, AttackConfig


class TestEpsilonEnvelope:
    def test_flat_envelope_without_mask(self):
        cfg = AttackConfig(epsilon=0.05, use_psychoacoustic_bound=True)
        engine = AdversarialEngine(cfg)

        wav_shape = (2, 44100)
        envelope = engine._build_epsilon_envelope(
            44100, masking_bound=None, wav_shape=wav_shape, sr=44100
        )
        assert envelope.shape == wav_shape
        np.testing.assert_allclose(envelope, 0.05)

    def test_envelope_from_mask(self):
        cfg = AttackConfig(epsilon=0.1, use_psychoacoustic_bound=True)
        engine = AdversarialEngine(cfg)

        n_frames, n_bins = 40, 221
        bound = np.random.rand(n_frames, n_bins).astype(np.float32) * 0.01
        wav_shape = (2, 44100)

        envelope = engine._build_epsilon_envelope(
            44100, masking_bound=bound, wav_shape=wav_shape, sr=44100
        )
        assert envelope.shape == wav_shape
        # Should never exceed epsilon
        assert envelope.max() <= cfg.epsilon + 1e-7

    def test_envelope_positive(self):
        cfg = AttackConfig(epsilon=0.02)
        engine = AdversarialEngine(cfg)

        bound = np.ones((20, 100), dtype=np.float32) * 0.005
        envelope = engine._build_epsilon_envelope(
            22050, masking_bound=bound, wav_shape=(1, 22050), sr=44100
        )
        assert np.all(envelope > 0)
