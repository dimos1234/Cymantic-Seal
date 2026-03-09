"""Adversarial perturbation engine targeting source-separation models.

Supports FGSM (single-step) and I-FGSM (iterative) attacks against
Demucs (HTDemucs) and, optionally, OpenUnmix / Spleeter.  Perturbations
are bounded by a per-time-frequency-tile psychoacoustic mask so they
remain imperceptible.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

DEMUCS_SR = 44100  # Demucs v4 native sample-rate


@dataclass
class AttackConfig:
    """Knobs for the adversarial optimisation loop."""

    method: Literal["fgsm", "ifgsm"] = "ifgsm"
    steps: int = 5
    epsilon: float = 0.008  # global fallback epsilon (linear amplitude)
    use_psychoacoustic_bound: bool = True
    device: str = "auto"
    model_name: str = "htdemucs"
    target_sources: list[str] = field(default_factory=lambda: ["vocals"])


class AdversarialEngine:
    """Generate gradient-guided perturbations that break source separation."""

    def __init__(self, config: AttackConfig | None = None):
        self.cfg = config or AttackConfig()
        self._device = self._resolve_device(self.cfg.device)
        self._model: torch.nn.Module | None = None
        self._training_length: int | None = None

    # ── model management ──────────────────────────────────────────

    @staticmethod
    def _resolve_device(hint: str) -> torch.device:
        if hint == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(hint)

    def _load_model(self) -> torch.nn.Module:
        if self._model is not None:
            return self._model

        try:
            from demucs.pretrained import get_model
            from demucs.apply import BagOfModels
        except ImportError as exc:
            raise RuntimeError(
                "demucs is required for adversarial attacks.  "
                "Install it with:  pip install demucs"
            ) from exc

        logger.info("Loading Demucs model '%s' ...", self.cfg.model_name)
        model = get_model(self.cfg.model_name)
        if isinstance(model, BagOfModels):
            model = model.models[0]

        model.to(self._device)
        model.eval()

        seg = getattr(model, "segment", None)
        sr = getattr(model, "samplerate", DEMUCS_SR)
        if seg is not None:
            self._training_length = int(float(seg) * int(sr))
        else:
            self._training_length = int(10.0 * DEMUCS_SR)

        logger.info(
            "Model training_length = %d samples (%.2f s)",
            self._training_length,
            self._training_length / DEMUCS_SR,
        )

        self._model = model
        return model

    # ── loss function ─────────────────────────────────────────────

    @staticmethod
    def _separation_loss(
        separated: torch.Tensor,
        mixture: torch.Tensor,
        source_indices: list[int],
    ) -> torch.Tensor:
        """Loss that, when *maximised*, degrades the quality of the
        target source estimates.
        """
        loss = torch.tensor(0.0, device=separated.device)
        for idx in source_indices:
            stem = separated[:, idx]  # (batch, channels, samples)
            loss = loss - F.mse_loss(stem, mixture)
        return loss

    # ── core attack ───────────────────────────────────────────────

    def generate_perturbation(
        self,
        waveform: np.ndarray,
        masking_bound: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return an additive perturbation array (same shape as *waveform*)."""
        waveform = waveform.astype(np.float32)
        model = self._load_model()
        n_channels, n_samples = waveform.shape

        source_names = model.sources
        source_indices = [
            source_names.index(s) for s in self.cfg.target_sources if s in source_names
        ]
        if not source_indices:
            logger.warning(
                "None of the target sources %s found in model sources %s; "
                "defaulting to all sources.",
                self.cfg.target_sources,
                source_names,
            )
            source_indices = list(range(len(source_names)))

        eps_envelope = self._build_epsilon_envelope(
            n_samples, masking_bound, waveform.shape, sr=DEMUCS_SR
        )

        seg_len = self._training_length or int(10.0 * DEMUCS_SR)
        # Non-overlapping segments for speed (overlap adds quality but
        # doubles processing time — not worth it on CPU).
        hop = seg_len
        perturbation = np.zeros_like(waveform)
        weight = np.zeros(n_samples, dtype=np.float32)

        total_segments = max(1, (n_samples + hop - 1) // hop)
        seg_idx = 0
        t_start = time.time()
        steps = 1 if self.cfg.method == "fgsm" else self.cfg.steps

        logger.info(
            "Processing %d segments (%.1fs each, %s x%d) on %s ...",
            total_segments,
            seg_len / DEMUCS_SR,
            self.cfg.method.upper(),
            steps,
            self._device,
        )

        for start in range(0, n_samples, hop):
            seg_idx += 1
            end = min(start + seg_len, n_samples)
            seg_wav = waveform[:, start:end]
            seg_eps = eps_envelope[:, start:end]

            logger.info(
                "  Segment %d/%d (%.1fs - %.1fs) ...",
                seg_idx, total_segments,
                start / DEMUCS_SR, end / DEMUCS_SR,
            )
            seg_t = time.time()

            seg_pert = self._attack_segment(
                seg_wav, model, source_indices, seg_eps, seg_len
            )
            actual_len = end - start
            perturbation[:, start:end] += seg_pert[:, :actual_len]
            weight[start:end] += 1.0

            elapsed = time.time() - seg_t
            logger.info("    Done in %.1fs", elapsed)

        weight = np.maximum(weight, 1.0)
        perturbation /= weight[np.newaxis, :]

        total_elapsed = time.time() - t_start
        logger.info("All segments done in %.1fs", total_elapsed)

        return perturbation.astype(np.float32)

    # ── segment-level FGSM / I-FGSM ──────────────────────────────

    def _attack_segment(
        self,
        segment: np.ndarray,
        model: torch.nn.Module,
        source_indices: list[int],
        eps_envelope: np.ndarray,
        target_length: int,
    ) -> np.ndarray:
        """Run FGSM or I-FGSM on a single waveform segment.

        The segment is zero-padded to *target_length* (the model's
        ``training_length``) so that the Demucs forward pass never
        encounters a shape mismatch.
        """
        segment = segment.astype(np.float32)
        eps_envelope = eps_envelope.astype(np.float32)
        n_channels, orig_len = segment.shape

        if orig_len < target_length:
            pad_amount = target_length - orig_len
            segment = np.pad(segment, ((0, 0), (0, pad_amount)), mode="constant")
            eps_envelope = np.pad(
                eps_envelope, ((0, 0), (0, pad_amount)),
                mode="constant", constant_values=self.cfg.epsilon,
            )

        x = torch.from_numpy(segment).unsqueeze(0).to(self._device)  # (1, C, T)
        eps_t = torch.from_numpy(eps_envelope).unsqueeze(0).to(self._device)

        steps = 1 if self.cfg.method == "fgsm" else self.cfg.steps
        alpha = eps_t / max(steps, 1)

        delta = torch.zeros_like(x)

        for step_i in range(steps):
            delta.requires_grad_(True)
            adv = x + delta

            with torch.enable_grad():
                separated = model(adv)  # (1, n_sources, C, T)
                mixture_expanded = x    # (1, C, T)
                loss = self._separation_loss(
                    separated, mixture_expanded, source_indices
                )

            loss.backward()
            grad = delta.grad.detach()

            with torch.no_grad():
                delta = delta + alpha * grad.sign()
                delta = torch.clamp(delta, -eps_t, eps_t)
                delta = delta.detach()

            if steps > 1:
                logger.info("      Step %d/%d  loss=%.6f", step_i + 1, steps, loss.item())

        return delta.squeeze(0).cpu().numpy()

    # ── epsilon envelope ──────────────────────────────────────────

    def _build_epsilon_envelope(
        self,
        n_samples: int,
        masking_bound: np.ndarray | None,
        wav_shape: tuple[int, int],
        sr: int,
    ) -> np.ndarray:
        """Convert the STFT-domain masking bound to a per-sample envelope."""
        n_channels = wav_shape[0]

        if masking_bound is None or not self.cfg.use_psychoacoustic_bound:
            return np.full(wav_shape, self.cfg.epsilon, dtype=np.float32)

        n_frames, n_bins = masking_bound.shape
        if n_frames == 0:
            return np.full(wav_shape, self.cfg.epsilon, dtype=np.float32)

        frame_ms = 10.0
        hop = max(1, int(sr * frame_ms / 1000.0 * 0.5))

        # Use 20th percentile per frame so quiet passages get a much lower cap
        # (max over bins can still be high in quiet frames due to ATH).
        frame_cap = np.percentile(masking_bound, 20, axis=1)
        # Global ceiling: never allow more than this in any sample (reduces hiss in silence).
        envelope_ceiling = min(self.cfg.epsilon, 0.004)
        frame_env = np.minimum(frame_cap, envelope_ceiling)

        indices = np.arange(n_samples, dtype=np.float32) / hop
        indices = np.clip(indices, 0, n_frames - 1)
        lower = np.floor(indices).astype(int)
        upper = np.minimum(lower + 1, n_frames - 1)
        frac = indices - lower

        envelope = frame_env[lower] * (1 - frac) + frame_env[upper] * frac
        envelope = np.clip(envelope, 1e-7, self.cfg.epsilon).astype(np.float32)

        return np.broadcast_to(
            envelope[np.newaxis, :n_samples], (n_channels, n_samples)
        ).copy()
