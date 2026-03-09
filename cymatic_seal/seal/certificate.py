"""Forensic Seal Certificate — generation, serialisation, and hashing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cymatic_seal import __version__


@dataclass
class SealCertificate:
    """Machine-readable record of the sealing operation."""

    version: str = __version__
    algorithm: str = "cymatic-seal-ifgsm-v1"
    timestamp: str = ""
    artist: str = ""
    title: str = ""
    original_filename: str = ""
    sample_rate: int = 44100
    channels: int = 2
    duration_seconds: float = 0.0
    masking_config: dict[str, Any] = field(default_factory=dict)
    attack_config: dict[str, Any] = field(default_factory=dict)
    frequency_bands_masked: list[dict[str, float]] = field(default_factory=list)
    perturbation_stats: dict[str, float] = field(default_factory=dict)
    audio_hash_sha256: str = ""
    sealed_audio_hash_sha256: str = ""

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> SealCertificate:
        data = json.loads(text)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def _hash_audio(waveform: np.ndarray) -> str:
    """SHA-256 of the raw float32 sample bytes (in-memory fallback)."""
    return hashlib.sha256(waveform.astype(np.float32).tobytes()).hexdigest()


def hash_file(path: str | Path) -> str:
    """SHA-256 of the raw bytes of a file on disk."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _summarise_bands(
    freqs: np.ndarray,
    threshold: np.ndarray,
) -> list[dict[str, float]]:
    """Return a compact summary of which frequency regions were masked."""
    n_bins = len(freqs)
    if n_bins == 0:
        return []
    n_bands = min(25, n_bins)
    band_size = max(1, n_bins // n_bands)
    bands: list[dict[str, float]] = []
    for b in range(n_bands):
        lo = b * band_size
        hi = min(lo + band_size, n_bins)
        bands.append(
            {
                "freq_lo_hz": float(freqs[lo]),
                "freq_hi_hz": float(freqs[min(hi, n_bins - 1)]),
                "mean_threshold_amplitude": float(threshold[:, lo:hi].mean()),
                "max_threshold_amplitude": float(threshold[:, lo:hi].max()),
            }
        )
    return bands


def generate_certificate(
    original_waveform: np.ndarray,
    sealed_waveform: np.ndarray,
    perturbation: np.ndarray,
    freqs: np.ndarray,
    threshold: np.ndarray,
    sr: int,
    *,
    artist: str = "",
    title: str = "",
    original_filename: str = "",
    attack_cfg: dict[str, Any] | None = None,
    original_file_path: str | Path | None = None,
    sealed_file_path: str | Path | None = None,
    margin_db: float = -4.0,
) -> SealCertificate:
    """Build a :class:`SealCertificate` from the sealing artefacts.

    When *sealed_file_path* is provided the hash is computed from the
    file bytes on disk, which avoids PCM-quantisation mismatches during
    verification.
    """
    n_channels = sealed_waveform.shape[0]
    n_samples = sealed_waveform.shape[1]

    if original_file_path is not None:
        orig_hash = hash_file(original_file_path)
    else:
        orig_hash = _hash_audio(original_waveform)

    if sealed_file_path is not None:
        sealed_hash = hash_file(sealed_file_path)
    else:
        sealed_hash = _hash_audio(sealed_waveform)

    cert = SealCertificate(
        timestamp=datetime.now(timezone.utc).isoformat(),
        artist=artist,
        title=title,
        original_filename=original_filename,
        sample_rate=sr,
        channels=n_channels,
        duration_seconds=round(n_samples / sr, 4),
        masking_config={
            "frame_ms": 10,
            "hop_ratio": 0.5,
            "temporal_masking": True,
            "margin_db": margin_db,
        },
        attack_config=attack_cfg or {},
        frequency_bands_masked=_summarise_bands(freqs, threshold),
        perturbation_stats={
            "mean_abs": float(np.mean(np.abs(perturbation))),
            "max_abs": float(np.max(np.abs(perturbation))),
            "rms": float(np.sqrt(np.mean(perturbation.astype(np.float64) ** 2))),
        },
        audio_hash_sha256=orig_hash,
        sealed_audio_hash_sha256=sealed_hash,
    )
    return cert
