"""Full-spectrum psychoacoustic masking model.

Implements ISO 11172-3 / MPEG-1 Layer I inspired simultaneous masking with
a bark-scale spreading function, plus forward/backward temporal masking so
that perturbations near transients can be stronger.  The output is a
per-frame, per-bin maximum perturbation magnitude (in linear amplitude)
that stays below the threshold of audibility.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# ── constants ────────────────────────────────────────────────────────

FRAME_MS = 10  # analysis window length in milliseconds
HOP_RATIO = 0.5  # 50 % overlap
ATH_REF_SPL = 96.0  # dB SPL corresponding to 0 dBFS

# Absolute threshold of hearing (simplified Terhardt formula),
# evaluated lazily for a given FFT size / sample-rate pair.
_ath_cache: dict[tuple[int, int], NDArray] = {}


# ── helpers ──────────────────────────────────────────────────────────

def _hz_to_bark(hz: NDArray) -> NDArray:
    """Zwicker & Terhardt critical-band rate (Bark)."""
    return 13.0 * np.arctan(0.00076 * hz) + 3.5 * np.arctan((hz / 7500.0) ** 2)


def _absolute_threshold_of_hearing(freqs_hz: NDArray) -> NDArray:
    """ATH in dB SPL (Terhardt, 1979).  Clipped at 96 dB for DC/0 Hz."""
    f_khz = np.clip(freqs_hz, 20.0, 20000.0) / 1000.0
    ath = (
        3.64 * f_khz ** -0.8
        - 6.5 * np.exp(-0.6 * (f_khz - 3.3) ** 2)
        + 1e-3 * f_khz ** 4
    )
    return ath


def _spreading_function(dz: NDArray) -> NDArray:
    """Schroeder spreading function on the Bark scale.

    *dz* is the signed distance (masker bark − maskee bark).
    Returns attenuation in dB.
    """
    return 15.81 + 7.5 * (dz + 0.474) - 17.5 * np.sqrt(1.0 + (dz + 0.474) ** 2)


def _db_to_linear(db: NDArray) -> NDArray:
    return 10.0 ** (db / 20.0)


def _linear_to_db(lin: NDArray) -> NDArray:
    return 20.0 * np.log10(np.maximum(lin, 1e-12))


# ── public API ───────────────────────────────────────────────────────

def compute_masking_threshold(
    waveform: NDArray,
    sr: int = 44100,
    frame_ms: float = FRAME_MS,
    *,
    temporal: bool = True,
    margin_db: float = -4.0,
) -> tuple[NDArray, NDArray]:
    """Compute per-frame, per-frequency-bin masking threshold.

    Parameters
    ----------
    waveform : 1-D float array (mono, one channel at a time).
    sr : sample rate.
    frame_ms : analysis window in ms (default 10 ms).
    temporal : whether to apply forward/backward temporal masking.
    margin_db : safety margin below the masking threshold (negative =
        more conservative / quieter perturbation).

    Returns
    -------
    threshold : float32 array of shape ``(n_frames, n_bins)`` giving the
        maximum allowed *linear amplitude* of a perturbation in each
        time-frequency tile.
    freqs : 1-D array of centre frequencies for each bin.
    """
    if waveform.ndim != 1:
        raise ValueError(
            f"Expected 1-D mono waveform, got shape {waveform.shape}. "
            "Pass one channel at a time."
        )

    waveform = waveform.astype(np.float32)
    frame_len = int(sr * frame_ms / 1000.0)
    hop = max(1, int(frame_len * HOP_RATIO))
    n_fft = frame_len
    window = np.hanning(frame_len).astype(np.float32)

    # ── STFT ──────────────────────────────────────────────────────
    n_samples = waveform.shape[-1]
    n_frames = 1 + (n_samples - frame_len) // hop
    if n_frames < 1:
        n_frames = 1

    n_bins = n_fft // 2 + 1
    freqs = np.linspace(0, sr / 2, n_bins, dtype=np.float32)
    bark = _hz_to_bark(freqs)

    # ATH in dB SPL (shifted so 0 dBFS = ATH_REF_SPL dB SPL)
    ath_db = _absolute_threshold_of_hearing(freqs)

    spectra_db = np.empty((n_frames, n_bins), dtype=np.float32)

    for i in range(n_frames):
        start = i * hop
        seg = waveform[start : start + frame_len]
        if len(seg) < frame_len:
            seg = np.pad(seg, (0, frame_len - len(seg)))
        windowed = seg * window
        spectrum = np.abs(np.fft.rfft(windowed, n=n_fft))
        spectra_db[i] = _linear_to_db(spectrum / n_fft)

    # ── simultaneous masking ──────────────────────────────────────
    # For each frame, spread each bin's energy across neighbouring
    # Bark bands via the spreading function, then take the max as the
    # masking threshold at each bin.

    threshold_db = np.full_like(spectra_db, -120.0)

    # Pre-compute pair-wise Bark distances (n_bins × n_bins).
    # To keep memory reasonable for large FFTs, we operate in bands.
    n_bark_bands = 25
    bark_edges = np.linspace(0, 25, n_bark_bands + 1)

    band_idx = np.digitize(bark, bark_edges) - 1
    band_idx = np.clip(band_idx, 0, n_bark_bands - 1)

    # Average energy per Bark band per frame
    band_energy_db = np.full((n_frames, n_bark_bands), -120.0, dtype=np.float32)
    for b in range(n_bark_bands):
        mask = band_idx == b
        if mask.any():
            band_energy_db[:, b] = spectra_db[:, mask].max(axis=1)

    bark_centres = (bark_edges[:-1] + bark_edges[1:]) / 2.0

    for b_masker in range(n_bark_bands):
        dz = bark_centres - bark_centres[b_masker]
        spread_db = _spreading_function(dz)  # shape (n_bark_bands,)
        # masking contribution of this band to every other band
        contrib = band_energy_db[:, b_masker : b_masker + 1] + spread_db[np.newaxis, :]
        # Map band threshold back to bins
        for b_maskee in range(n_bark_bands):
            sel = band_idx == b_maskee
            if sel.any():
                threshold_db[:, sel] = np.maximum(
                    threshold_db[:, sel],
                    contrib[:, b_maskee : b_maskee + 1],
                )

    # Combine with ATH — threshold can never be below ATH
    threshold_db = np.maximum(threshold_db, ath_db[np.newaxis, :] - ATH_REF_SPL)

    # ── temporal masking (forward + backward) ─────────────────────
    if temporal and n_frames > 1:
        decay_per_frame = 3.0  # dB decay per frame (~3 dB / 5 ms at 50 % hop)
        # forward masking: each frame's threshold can raise the *next* frame
        for i in range(1, n_frames):
            carried = threshold_db[i - 1] - decay_per_frame
            threshold_db[i] = np.maximum(threshold_db[i], carried)
        # backward (pre-) masking: weaker, 1-frame look-ahead
        for i in range(n_frames - 2, -1, -1):
            carried = threshold_db[i + 1] - decay_per_frame * 2.0
            threshold_db[i] = np.maximum(threshold_db[i], carried)

    # ── apply safety margin and convert to linear amplitude ───────
    threshold_db += margin_db
    threshold_linear = _db_to_linear(threshold_db).astype(np.float32)

    return threshold_linear, freqs
