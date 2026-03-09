"""Audio I/O: load, save, resample, and normalize waveforms."""

from __future__ import annotations

import numpy as np
import soundfile as sf

INTERNAL_SR = 44100


def load_audio(
    path: str,
    sr: int = INTERNAL_SR,
    mono: bool = False,
) -> tuple[np.ndarray, int]:
    """Load an audio file, resample to *sr*, and return (waveform, sr).

    Returns waveform shaped (channels, samples) as float32.
    If *mono* is True the channels are averaged to a single row.
    """
    data, file_sr = sf.read(path, dtype="float32", always_2d=True)

    if data.size == 0:
        raise ValueError(f"Audio file is empty or unreadable: {path}")

    # soundfile returns (samples, channels) -- transpose to (channels, samples)
    data = data.T.astype(np.float32)

    if file_sr != sr:
        data = resample(data, file_sr, sr)

    if mono and data.shape[0] > 1:
        data = data.mean(axis=0, keepdims=True).astype(np.float32)

    return data, sr


def save_audio(
    path: str,
    waveform: np.ndarray,
    sr: int = INTERNAL_SR,
    subtype: str | None = None,
) -> None:
    """Save a (channels, samples) waveform to *path*.

    Format is inferred from the file extension.  *subtype* lets the caller
    choose e.g. ``"PCM_24"`` for 24-bit WAV.
    """
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]
    waveform = waveform.astype(np.float32)
    # soundfile expects (samples, channels)
    sf.write(path, waveform.T, sr, subtype=subtype)


def resample(
    waveform: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    """Resample a (channels, samples) array using polyphase filtering."""
    from scipy.signal import resample_poly
    import math

    if orig_sr == target_sr:
        return waveform

    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError(f"Sample rates must be positive, got {orig_sr} / {target_sr}")

    gcd = math.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd

    return np.stack(
        [resample_poly(ch, up, down).astype(np.float32) for ch in waveform]
    )


def lowpass_perturbation(
    perturbation: np.ndarray,
    sr: int = INTERNAL_SR,
    cutoff_hz: float = 8000.0,
    order: int = 2,
) -> np.ndarray:
    """Apply a zero-phase Butterworth low-pass to remove harsh high frequencies.

    Operates per-channel on a (channels, samples) perturbation array.
    Uses sosfiltfilt for zero-phase filtering so timing is not shifted.
    """
    from scipy.signal import butter, sosfiltfilt

    nyquist = sr / 2.0
    if cutoff_hz >= nyquist:
        return perturbation
    sos = butter(order, cutoff_hz / nyquist, btype="low", output="sos")
    filtered = np.stack(
        [sosfiltfilt(sos, ch).astype(np.float32) for ch in perturbation]
    )
    return filtered


def normalize(waveform: np.ndarray, peak: float = 0.95) -> np.ndarray:
    """Peak-normalize so the loudest sample reaches +/-*peak*."""
    mx = np.max(np.abs(waveform))
    if mx < 1e-8:
        return waveform.astype(np.float32)
    return (waveform * np.float32(peak / mx)).astype(np.float32)
