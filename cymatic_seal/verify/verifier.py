"""Seal verification: confirm that a file matches its Forensic Seal Certificate."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from cymatic_seal.seal.certificate import SealCertificate, hash_file


@dataclass
class VerificationResult:
    """Outcome of a seal-verification check."""

    verified: bool
    reason: str
    certificate: SealCertificate | None = None


def verify_seal(
    audio_path: str | Path,
    certificate: SealCertificate | str | Path,
) -> VerificationResult:
    """Verify that *audio_path* matches the sealed-audio hash in *certificate*.

    Parameters
    ----------
    audio_path : path to the audio file to check.
    certificate : a :class:`SealCertificate`, a JSON string, or a path to
        a ``.json`` certificate file.

    Returns
    -------
    VerificationResult with ``verified=True`` if the hashes match.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        return VerificationResult(
            verified=False,
            reason=f"Audio file not found: {audio_path}",
        )

    # ── resolve certificate ───────────────────────────────────────
    if isinstance(certificate, (str, Path)):
        cert_path = Path(certificate)
        if cert_path.suffix == ".json" and cert_path.exists():
            text = cert_path.read_text(encoding="utf-8")
            cert = SealCertificate.from_json(text)
        elif cert_path.suffix == ".json" and not cert_path.exists():
            return VerificationResult(
                verified=False,
                reason=f"Certificate file not found: {cert_path}",
            )
        else:
            try:
                cert = SealCertificate.from_json(str(certificate))
            except json.JSONDecodeError as exc:
                return VerificationResult(
                    verified=False,
                    reason=f"Invalid certificate JSON: {exc}",
                )
    else:
        cert = certificate

    if not cert.sealed_audio_hash_sha256:
        return VerificationResult(
            verified=False,
            reason="Certificate does not contain a sealed-audio hash.",
            certificate=cert,
        )

    # ── hash the file on disk ─────────────────────────────────────
    computed_hash = hash_file(str(audio_path))

    if computed_hash == cert.sealed_audio_hash_sha256:
        return VerificationResult(
            verified=True,
            reason="Audio hash matches the certificate. This file is sealed.",
            certificate=cert,
        )

    return VerificationResult(
        verified=False,
        reason=(
            "Hash mismatch -- the audio file does not match the certificate. "
            "The file may have been re-encoded, modified, or is not the sealed version."
        ),
        certificate=cert,
    )
