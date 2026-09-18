"""Separator contract: ``separate(mixture) -> [2, time]`` (PRD §11.2).

Accepts input shape ``[time]`` or ``[1, time]``; always returns ``[2, time]``
without modifying the input in place.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch


def _as_mono_1d(mixture: torch.Tensor) -> torch.Tensor:
    """Validate and flatten separator input to 1-D mono.

    Args:
        mixture: Tensor of shape ``[time]`` or ``[1, time]``.

    Returns:
        Cloned 1-D ``float32`` tensor.

    Raises:
        ValueError: If the shape is not ``[time]`` or ``[1, time]``.
    """
    wav = torch.as_tensor(mixture, dtype=torch.float32)
    if wav.ndim == 1:
        if wav.numel() == 0:
            raise ValueError("BaseSeparator input must be non-empty.")
        return wav.clone()
    if wav.ndim == 2 and wav.shape[0] == 1:
        if wav.numel() == 0:
            raise ValueError("BaseSeparator input must be non-empty.")
        return wav.reshape(-1).clone()
    raise ValueError(
        f"BaseSeparator input must have shape [time] or [1, time], got {tuple(wav.shape)}."
    )


class BaseSeparator(ABC):
    """Abstract separator matching PRD §11.2.

    Attributes:
        name: Separator id (``mixture``, ``nmf_enrolled``, ``neural``, ``oracle``).
        sample_rate_hz: Native sample rate (16 kHz contract).
        num_sources: Always 2 in v1.
    """

    name: str = "base"
    sample_rate_hz: int = 16000
    num_sources: int = 2

    @abstractmethod
    def separate(self, mixture: torch.Tensor) -> torch.Tensor:
        """Separate a mono mixture into two tracks.

        Args:
            mixture: Tensor of shape ``[time]`` or ``[1, time]``.

        Returns:
            Tensor of shape ``[2, time]``; input never modified in place.
        """
        ...

    def _check_output(self, output: torch.Tensor, length: int) -> torch.Tensor:
        """Validate separator output shape ``[2, time]``.

        Args:
            output: Candidate output tensor.
            length: Expected time dimension.

        Returns:
            The validated output tensor.

        Raises:
            ValueError: If shape is not ``[2, length]``.
        """
        if output.shape != torch.Size([2, length]):
            raise ValueError(
                f"Separator output must have shape [2, {length}], got {tuple(output.shape)}."
            )
        return output


class Separator(BaseSeparator):
    """Backwards-compatible alias for the abstract separator protocol."""

    @abstractmethod
    def separate(self, mixture: torch.Tensor) -> torch.Tensor:
        """Separate a mono mixture into two tracks (see :class:`BaseSeparator`)."""
        ...


__all__ = ["BaseSeparator", "Separator", "_as_mono_1d"]
