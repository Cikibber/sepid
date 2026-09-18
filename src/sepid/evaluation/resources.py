"""Resource profiling: RTF, peak RAM/VRAM, model size (Epic E, PRD §14.3)."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ResourceReport:
    """Measured resource costs for one file set.

    Attributes:
        audio_seconds: Total input audio duration.
        processing_seconds: Wall-clock inference time (excl. download/setup).
        real_time_factor: processing_seconds / audio_seconds.
        device: Hardware tag (e.g. ``"cpu"``).
    """

    audio_seconds: float
    processing_seconds: float
    real_time_factor: float
    device: str = "cpu"


def profile(fn: Callable[[], None], audio_seconds: float, device: str = "cpu") -> ResourceReport:
    """Time a callable and report its real-time factor.

    Args:
        fn: Work to measure (warm-up excluded by caller).
        audio_seconds: Input audio duration in seconds.
        device: Hardware tag.

    Returns:
        :class:`ResourceReport` with measured RTF.
    """
    start = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - start
    rtf = elapsed / max(float(audio_seconds), 1e-12)
    return ResourceReport(
        audio_seconds=float(audio_seconds),
        processing_seconds=elapsed,
        real_time_factor=rtf,
        device=device,
    )


__all__ = ["ResourceReport", "profile"]
