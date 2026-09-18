"""M3 oracle mask control: IRM/IBM upper bound (PRD §8.1 M3, FR-6).

Uses clean references to build an ideal ratio mask (IRM, ``mask=soft``) or
ideal binary mask (IBM, ``mask=binary``), applies it to the mixture STFT, and
reconstructs two tracks via iSTFT. Non-deployable analysis control: it
requires references, so invoking :meth:`separate` without them raises.
Frozen STFT: ``N_fft=512``, ``hop_length=128``, Hann window.
"""
from __future__ import annotations

import torch

from sepid.audio.preprocessing import HOP_LENGTH, N_FFT, compute_istft, compute_stft
from sepid.separators.base import BaseSeparator, _as_mono_1d


class OracleMaskError(RuntimeError):
    """Raised when the oracle separator is misused (no references, demo mode)."""


class OracleMaskSeparator(BaseSeparator):
    """Ideal-mask oracle control (labeled non-deployable in every report).

    Attributes:
        name: Always ``"oracle"``.
        sample_rate_hz: Native sample rate (16 kHz contract).
        num_sources: Always 2 in v1.
        mask: ``"soft"`` for IRM (magnitude ratio) or ``"binary"`` for IBM
            (winner-takes-all per TF bin).
        n_fft: Frozen FFT size. hop_length: Frozen hop.
    """

    name: str = "oracle"
    sample_rate_hz: int = 16000
    num_sources: int = 2

    def __init__(self, mask: str = "soft", n_fft: int = N_FFT, hop_length: int = HOP_LENGTH) -> None:
        """Create the oracle separator with a frozen mask rule.

        Args:
            mask: ``"soft"`` (IRM) or ``"binary"`` (IBM).
            n_fft: FFT size (default 512).
            hop_length: Hop length (default 128).

        Raises:
            ValueError: For an unknown mask id.
        """
        if mask not in ("soft", "binary"):
            raise ValueError(f"Oracle mask must be 'soft' (IRM) or 'binary' (IBM), got {mask!r}.")
        self.mask: str = mask
        self.n_fft: int = int(n_fft)
        self.hop_length: int = int(hop_length)

    def separate(
        self,
        mixture: torch.Tensor,
        references: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Separate with an ideal mask built from clean references.

        Args:
            mixture: Tensor of shape ``[time]`` or ``[1, time]``.
            references: Tensor of shape ``[2, time]`` with clean sources.
                Required; ``None`` raises :class:`OracleMaskError` so the
                oracle can never run in reference-free demo mode.

        Returns:
            Tensor of shape ``[2, time]`` with masked reconstructions.

        Raises:
            OracleMaskError: If ``references`` is ``None`` or malformed.
        """
        wav = _as_mono_1d(mixture)
        if references is None:
            raise OracleMaskError(
                "M3 oracle requires clean references and is non-deployable; "
                "refusing to run in reference-free mode."
            )
        refs = torch.as_tensor(references, dtype=torch.float32)
        if refs.shape != torch.Size([2, wav.numel()]):
            raise OracleMaskError(
                f"M3 oracle references must have shape [2, {wav.numel()}], "
                f"got {tuple(refs.shape)}."
            )
        mix_spec = compute_stft(wav, n_fft=self.n_fft, hop_length=self.hop_length)
        mag0 = torch.abs(compute_stft(refs[0], n_fft=self.n_fft, hop_length=self.hop_length))
        mag1 = torch.abs(compute_stft(refs[1], n_fft=self.n_fft, hop_length=self.hop_length))
        denom = mag0 + mag1
        if self.mask == "soft":
            m0 = torch.where(denom > 0, mag0 / denom.clamp_min(1e-12), torch.full_like(denom, 0.5))
            m1 = 1.0 - m0
        else:  # IBM: winner-takes-all, ties broken toward source 0.
            m0 = (mag0 >= mag1).to(mix_spec.real.dtype)
            m1 = 1.0 - m0
        est0 = compute_istft(mix_spec * m0.to(mix_spec.dtype), length=wav.numel())
        est1 = compute_istft(mix_spec * m1.to(mix_spec.dtype), length=wav.numel())
        out = torch.stack([est0, est1], dim=0)
        return self._check_output(out, wav.numel())


__all__ = ["OracleMaskError", "OracleMaskSeparator"]
