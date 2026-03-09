"""End-to-end seal pipeline: audio in → sealed audio + certificate out."""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from cymatic_seal.audio import load_audio, lowpass_perturbation, save_audio
from cymatic_seal.psychoacoustics import compute_masking_threshold
from cymatic_seal.seal.certificate import (
    SealCertificate,
    generate_certificate,
)

logger = logging.getLogger(__name__)


def seal_audio(
    input_path: str | Path,
    output_path: str | Path | None = None,
    certificate_path: str | Path | None = None,
    *,
    artist: str = "",
    title: str = "",
    method: str = "ifgsm",
    steps: int = 5,
    epsilon: float = 0.008,
    device: str = "auto",
    model_name: str = "htdemucs",
    target_sources: list[str] | None = None,
    margin_db: float = -4.0,
    lowpass_cutoff_hz: float = 6000.0,
) -> tuple[np.ndarray, SealCertificate]:
    """Run the complete Cymatic Seal pipeline on an audio file.

    Returns the sealed waveform ``(channels, samples)`` and the
    certificate object.  If *output_path* / *certificate_path* are
    given, the results are also written to disk.
    """
    input_path = Path(input_path)
    if target_sources is None:
        target_sources = ["vocals"]

    logger.info("Loading audio from %s", input_path)
    waveform, sr = load_audio(str(input_path))

    # ── 1. psychoacoustic masking bound (per channel, merge later) ─
    logger.info("Computing psychoacoustic masking threshold …")
    # Process each channel and take the *minimum* threshold (conservative)
    thresholds = []
    for ch in range(waveform.shape[0]):
        thr, freqs = compute_masking_threshold(
            waveform[ch], sr=sr, margin_db=margin_db
        )
        thresholds.append(thr)
    masking_bound = np.min(np.stack(thresholds), axis=0)

    # ── 2. adversarial perturbation ───────────────────────────────
    from cymatic_seal.adversarial.engine import AdversarialEngine, AttackConfig

    attack_cfg = AttackConfig(
        method=method,  # type: ignore[arg-type]
        steps=steps,
        epsilon=epsilon,
        use_psychoacoustic_bound=True,
        device=device,
        model_name=model_name,
        target_sources=target_sources,
    )
    engine = AdversarialEngine(attack_cfg)
    logger.info(
        "Generating adversarial perturbation (%s, %d steps) …",
        method,
        steps,
    )
    perturbation = engine.generate_perturbation(waveform, masking_bound)

    # ── 2b. low-pass filter perturbation to remove harsh high frequencies ─
    logger.info("Applying %.0f Hz low-pass to perturbation …", lowpass_cutoff_hz)
    perturbation = lowpass_perturbation(perturbation, sr=sr, cutoff_hz=lowpass_cutoff_hz)

    # ── 3. apply perturbation ─────────────────────────────────────
    sealed = np.clip(waveform + perturbation, -1.0, 1.0).astype(np.float32)

    # ── 4. save sealed audio first (so certificate can hash the file) ─
    saved_output: Path | None = None
    if output_path is not None:
        saved_output = Path(output_path)
        saved_output.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Saving sealed audio to %s", saved_output)
        save_audio(str(saved_output), sealed, sr=sr)

    # ── 5. certificate ────────────────────────────────────────────
    logger.info("Generating Forensic Seal Certificate …")
    cert = generate_certificate(
        original_waveform=waveform,
        sealed_waveform=sealed,
        perturbation=perturbation,
        freqs=freqs,
        threshold=masking_bound,
        sr=sr,
        artist=artist,
        title=title,
        original_filename=input_path.name,
        attack_cfg={
            k: v
            for k, v in asdict(attack_cfg).items()
            if k != "device"
        },
        original_file_path=str(input_path),
        sealed_file_path=str(saved_output) if saved_output else None,
        margin_db=margin_db,
    )

    # ── 6. save certificate ──────────────────────────────────────
    if certificate_path is not None:
        certificate_path = Path(certificate_path)
        certificate_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Saving certificate to %s", certificate_path)
        certificate_path.write_text(cert.to_json(), encoding="utf-8")

    logger.info("Seal complete.")
    return sealed, cert
