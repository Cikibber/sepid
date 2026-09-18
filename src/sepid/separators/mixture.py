"""M0 mixture baseline: passthrough adapter (PRD §8.1 M0).

Duplicates the input mixture as both outputs. Accepts ``[time]`` or
``[1, time]`` and always returns ``[2, time]``.
"""
from __future__ import annotations

import torch

from sepid.separators.base import BaseSeparator, _as_mono_1d


class MixtureBaseline(BaseSeparator):
    """Unprocessed-mixture baseline (lower bound, PRD §8.1 M0)."""

    name: str = "mixture"
    sample_rate_hz: int = 16000
    num_sources: int = 2

    def separate(self, mixture: torch.Tensor) -> torch.Tensor:
        """Return ``[mixture, mixture]`` without modifying the input.

        Args:
            mixture: Tensor of shape ``[time]`` or ``[1, time]``.

        Returns:
            Tensor of shape ``[2, time]``.
        """
        wav = _as_mono_1d(mixture)
        out = torch.stack([wav, wav.clone()], dim=0)
        return self._check_output(out, wav.numel())


__all__ = ["MixtureBaseline"]
