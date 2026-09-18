"""Controlled mixing toolkit: gains/offsets/seeds + clipping checks (FR-13)."""
from __future__ import annotations

import numpy as np


def mix_at_level_ratio(source_1: np.ndarray, source_2: np.ndarray, ratio_db: float) -> np.ndarray:
    """Mix two sources at a given source-to-source level ratio.

    Args:
        source_1: First source waveform (1-D).
        source_2: Second source waveform (1-D, gain-adjusted).
        ratio_db: ``20*log10(rms1/rms2)`` target for source 1 vs source 2.

    Returns:
        Peak-protected mixture (no per-output loudness manipulation beyond
        clipping protection).
    """
    s1 = np.asarray(source_1, dtype=np.float64)
    s2 = np.asarray(source_2, dtype=np.float64)
    n = min(s1.shape[0], s2.shape[0])
    s1, s2 = s1[:n], s2[:n]
    rms1 = float(np.sqrt(np.mean(s1**2))) or 1e-12
    rms2 = float(np.sqrt(np.mean(s2**2))) or 1e-12
    target = 10.0 ** (float(ratio_db) / 20.0)
    s2_scaled = s2 * (rms1 / (rms2 * target + 1e-12))
    mix = s1 + s2_scaled
    peak = float(np.max(np.abs(mix))) or 1.0
    if peak > 1.0:
        mix = mix / peak
    return mix.astype(np.float32)


__all__ = ["mix_at_level_ratio"]
