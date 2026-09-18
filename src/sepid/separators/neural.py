"""M2 frozen neural separator adapter skeleton (Epic C)."""
from __future__ import annotations

import torch


class NeuralSeparator:
    """Frozen pretrained neural separator (checkpoint frozen in Phase 0)."""

    name: str = "neural"
    sample_rate_hz: int = 16000
    num_sources: int = 2

    def separate(self, mixture: torch.Tensor) -> torch.Tensor:
        """Run frozen checkpoint inference (not yet implemented).

        Args:
            mixture: 1-D mono waveform.

        Raises:
            NotImplementedError: Epic C work item.
        """
        raise NotImplementedError("M2 neural adapter lands in Epic C.")


__all__ = ["NeuralSeparator"]
