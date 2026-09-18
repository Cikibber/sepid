"""Compatibility shim: ``src/preprocessing.py`` re-exports canonical DSP + plots.

Canonical DSP: :mod:`sepid.audio.preprocessing` (FR-2).
Canonical plots: :mod:`sepid.audio.visualization` (FR-3).
"""
from __future__ import annotations

from sepid.audio.preprocessing import (
    HOP_LENGTH,
    N_FFT,
    PRE_EMPHASIS_COEF,
    TARGET_SAMPLE_RATE_HZ,
    check_clipping,
    compute_istft,
    compute_stft,
    de_emphasis,
    ensure_16k_mono,
    pre_emphasis,
    pre_emphasis_frequency_response,
    resample_audio,
    stft_roundtrip_error,
    to_mono,
)
from sepid.audio.visualization import (
    DEFAULT_FIGURE_DIR,
    figure_path,
    plot_before_after,
    plot_spectrogram,
    plot_waveform,
)

__all__ = [
    "DEFAULT_FIGURE_DIR",
    "HOP_LENGTH",
    "N_FFT",
    "PRE_EMPHASIS_COEF",
    "TARGET_SAMPLE_RATE_HZ",
    "check_clipping",
    "compute_istft",
    "compute_stft",
    "de_emphasis",
    "ensure_16k_mono",
    "figure_path",
    "plot_before_after",
    "plot_spectrogram",
    "plot_waveform",
    "pre_emphasis",
    "pre_emphasis_frequency_response",
    "resample_audio",
    "stft_roundtrip_error",
    "to_mono",
]
