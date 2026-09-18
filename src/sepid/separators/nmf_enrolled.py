"""M1 enrolled NMF separator skeleton (Epic C, classical DSP baseline)."""
from __future__ import annotations

import torch


class EnrolledNMFSeparator:
    """Speaker-enrolled semi-supervised NMF separator (frozen hyperparams)."""

    name: str = "nmf_enrolled"
    sample_rate_hz: int = 16000
    num_sources: int = 2

    def separate(self, mixture: torch.Tensor) -> torch.Tensor:
        """Separate with frozen speaker dictionaries (not yet implemented).

        Args:
            mixture: 1-D mono waveform.

        Raises:
            NotImplementedError: Epic C work item.
        """
        raise NotImplementedError("M1 enrolled NMF lands in Epic C.")


__all__ = ["EnrolledNMFSeparator"]
