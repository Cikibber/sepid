"""M2 frozen neural separator: Asteroid ConvTasNet adapter (PRD §8.3, FR-5).

Wraps ``JorisCos/ConvTasNet_Libri2Mix_sepclean_16k`` at a pinned revision.
CPU inference only (GPU optional upstream, never required). Validates the
16 kHz contract and restores the exact input length via pad/crop. The model
runs in eval mode under ``torch.no_grad``; weights are never fine-tuned.
"""
from __future__ import annotations

import torch

from sepid.separators.base import BaseSeparator, _as_mono_1d

#: Frozen M2 checkpoint (PRD §8.3 neural baseline contract).
CHECKPOINT_ID: str = "JorisCos/ConvTasNet_Libri2Mix_sepclean_16k"

#: Pinned Hub revision (immutable commit; see docs/MODEL_CARD.md).
CHECKPOINT_REVISION: str = "e1ef95ab7a037950f3a606b9a56760cf94701d3d"

#: Native sample-rate / source-count contract.
EXPECTED_SAMPLE_RATE_HZ: int = 16000
EXPECTED_NUM_SOURCES: int = 2


class NeuralSeparatorError(RuntimeError):
    """Raised when the M2 adapter contract is violated."""


class NeuralSeparator(BaseSeparator):
    """Frozen pretrained ConvTasNet separator (PRD §8.1 M2).

    Attributes:
        name: Always ``"neural"``.
        sample_rate_hz: Native sample rate (16 kHz contract).
        num_sources: Always 2 in v1.
        checkpoint_id: Hub id. revision: Pinned commit.
    """

    name: str = "neural"
    sample_rate_hz: int = EXPECTED_SAMPLE_RATE_HZ
    num_sources: int = EXPECTED_NUM_SOURCES

    def __init__(
        self,
        checkpoint_id: str = CHECKPOINT_ID,
        revision: str = CHECKPOINT_REVISION,
        device: str = "cpu",
        sample_rate_hz: int = EXPECTED_SAMPLE_RATE_HZ,
    ) -> None:
        """Create the adapter (weights load lazily on first :meth:`separate`).

        Args:
            checkpoint_id: Hub checkpoint id (must equal the frozen id).
            revision: Pinned Hub revision (must equal the frozen revision).
            device: Inference device; only ``"cpu"`` is supported (FR-5).
            sample_rate_hz: Declared input rate; must be 16 kHz.

        Raises:
            NeuralSeparatorError: For non-CPU device, wrong rate, or a
                non-frozen checkpoint/revision.
        """
        if device != "cpu":
            raise NeuralSeparatorError(
                f"M2 supports CPU inference only in v1, got device={device!r}."
            )
        if int(sample_rate_hz) != EXPECTED_SAMPLE_RATE_HZ:
            raise NeuralSeparatorError(
                f"M2 expects {EXPECTED_SAMPLE_RATE_HZ} Hz input, got {sample_rate_hz}."
            )
        if checkpoint_id != CHECKPOINT_ID or revision != CHECKPOINT_REVISION:
            raise NeuralSeparatorError(
                "M2 checkpoint is frozen: refusing "
                f"{checkpoint_id!r}@{revision!r}; expected "
                f"{CHECKPOINT_ID!r}@{CHECKPOINT_REVISION!r}. "
                "Switching requires a dated decision-log entry and full rerun."
            )
        self.checkpoint_id: str = checkpoint_id
        self.revision: str = revision
        self.device: str = device
        self._model: torch.nn.Module | None = None

    def _load(self) -> torch.nn.Module:
        """Load the frozen checkpoint once (CPU, eval mode).

        Returns:
            The eval-mode ConvTasNet module.

        Raises:
            NeuralSeparatorError: If asteroid is missing or the loaded model
                violates the 2-source contract.
        """
        if self._model is not None:
            return self._model
        try:
            from asteroid.models import ConvTasNet
        except ImportError as exc:
            raise NeuralSeparatorError(
                "M2 requires the `asteroid` package: pip install asteroid."
            ) from exc
        model = ConvTasNet.from_pretrained(f"{self.checkpoint_id}@{self.revision}")
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self._model = model.to(self.device)
        return self._model

    @property
    def num_parameters(self) -> int:
        """Count trainable-architecture parameters (loads weights if needed)."""
        return sum(p.numel() for p in self._load().parameters())

    def separate(self, mixture: torch.Tensor) -> torch.Tensor:
        """Run frozen inference, restoring the exact input length.

        Args:
            mixture: Tensor of shape ``[time]`` or ``[1, time]`` at 16 kHz.

        Returns:
            Tensor of shape ``[2, time]`` with ``time`` equal to the input
            length (automatic pad/crop around the model call).
        """
        wav = _as_mono_1d(mixture).to(self.device)
        length = wav.numel()
        model = self._load()
        with torch.no_grad():
            out = model(wav.reshape(1, -1))
        est = torch.as_tensor(out, dtype=torch.float32).reshape(2, -1)
        if est.shape[0] != EXPECTED_NUM_SOURCES:
            raise NeuralSeparatorError(
                f"M2 must emit {EXPECTED_NUM_SOURCES} sources, got {est.shape[0]}."
            )
        if est.shape[1] < length:
            est = torch.nn.functional.pad(est, (0, length - est.shape[1]))
        else:
            est = est[:, :length]
        return self._check_output(est.cpu(), length)


__all__ = [
    "CHECKPOINT_ID",
    "CHECKPOINT_REVISION",
    "EXPECTED_NUM_SOURCES",
    "EXPECTED_SAMPLE_RATE_HZ",
    "NeuralSeparator",
    "NeuralSeparatorError",
]
