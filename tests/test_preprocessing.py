"""Unit tests for DSP preprocessing (FR-2): FIR response + STFT/iSTFT roundtrip.

Validates:
  1. Pre-emphasis FIR ``y[n] = x[n] - 0.97 * x[n-1]`` time-domain behavior,
     impulse response, and closed-form frequency response.
  2. STFT (N_fft=512, hop=128, Hann) / iSTFT reconstruction error < 1e-4.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make `src/` layout importable without requiring an editable install.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sepid.audio.preprocessing import (
    HOP_LENGTH,
    N_FFT,
    PRE_EMPHASIS_COEF,
    compute_istft,
    compute_stft,
    de_emphasis,
    pre_emphasis,
    pre_emphasis_frequency_response,
)

TOL_RECONSTRUCTION: float = 1e-4


def test_pre_emphasis_time_domain_known_values() -> None:
    """Pre-emphasis must match y[n] = x[n] - 0.97 * x[n-1] on a known vector."""
    x = np.array([1.0, 2.0, 3.0, 0.5], dtype=np.float32)
    expected = np.array(
        [1.0, 2.0 - 0.97 * 1.0, 3.0 - 0.97 * 2.0, 0.5 - 0.97 * 3.0],
        dtype=np.float64,
    )
    y = pre_emphasis(x, coef=0.97)
    assert y.shape == x.shape
    assert y.dtype == np.float32
    np.testing.assert_allclose(y, expected, rtol=1e-6, atol=1e-7)


def test_pre_emphasis_impulse_response() -> None:
    """Impulse in -> [1, -coef, 0, ...]: direct FIR tap check."""
    n = 64
    x = np.zeros(n, dtype=np.float32)
    x[0] = 1.0
    y = pre_emphasis(x, coef=PRE_EMPHASIS_COEF)
    assert float(y[0]) == 1.0
    assert abs(float(y[1]) - (-PRE_EMPHASIS_COEF)) < 1e-7
    np.testing.assert_allclose(y[2:], np.zeros(n - 2), atol=1e-8)


def test_pre_emphasis_frequency_response_endpoints() -> None:
    """Closed-form H(w) = 1 - coef*e^{-jw}: DC -> 0.03, Nyquist -> 1.97."""
    freqs = np.array([0.0, np.pi], dtype=np.float64)
    h = pre_emphasis_frequency_response(freqs, coef=0.97)
    assert h.shape == (2,)
    np.testing.assert_allclose(h[0], 1.0 - 0.97, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(h[1], 1.0 + 0.97, rtol=1e-12, atol=1e-12)
    assert abs(abs(h[1]) - 1.97) < 1e-12


def test_pre_emphasis_frequency_response_matches_fir_dft() -> None:
    """Closed-form response must match the DFT of FIR taps [1, -coef]."""
    rng = np.random.default_rng(0)
    freqs = np.linspace(0.0, np.pi, 257)
    h_closed = pre_emphasis_frequency_response(freqs, coef=PRE_EMPHASIS_COEF)
    taps = np.array([1.0, -PRE_EMPHASIS_COEF])
    # Direct DTFT of a 2-tap FIR at the same frequencies.
    n_grid = freqs[:, None] * np.arange(2)[None, :]
    h_direct = (taps[None, :] * np.exp(-1j * n_grid)).sum(axis=1)
    np.testing.assert_allclose(h_closed, h_direct, rtol=1e-12, atol=1e-12)
    assert rng is not None  # keep deterministic seed pattern explicit


def test_de_emphasis_inverts_pre_emphasis() -> None:
    """de_emphasis(pre_emphasis(x)) must recover x to float precision."""
    rng = np.random.default_rng(7)
    x = rng.standard_normal(8000).astype(np.float32) * 0.3
    rec = de_emphasis(pre_emphasis(x, coef=0.97), coef=0.97)
    assert float(np.max(np.abs(rec - x))) < 1e-5


def test_stft_params_and_shape() -> None:
    """STFT must use N_fft=512 / hop=128 / Hann with 257 frequency bins."""
    assert N_FFT == 512
    assert HOP_LENGTH == 128
    sr = 16000
    t = np.arange(sr, dtype=np.float32) / sr
    x = (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    spec = compute_stft(x, n_fft=N_FFT, hop_length=HOP_LENGTH, window="hann")
    assert spec.shape[0] == N_FFT // 2 + 1 == 257
    assert spec.shape[1] > 0


def test_stft_istft_reconstruction_error_below_threshold() -> None:
    """STFT/iSTFT roundtrip max-abs error must be < 1e-4 (sine + noise)."""
    sr = 16000
    t = np.arange(sr, dtype=np.float64) / sr
    rng = np.random.default_rng(42)
    x = (
        0.5 * np.sin(2.0 * np.pi * 440.0 * t)
        + 0.25 * np.sin(2.0 * np.pi * 880.0 * t)
        + 0.05 * rng.standard_normal(sr)
    ).astype(np.float32)
    spec = compute_stft(x, n_fft=512, hop_length=128, window="hann")
    rec = compute_istft(spec, length=int(x.shape[0]), n_fft=512, hop_length=128)
    rec_np = rec.numpy() if hasattr(rec, "numpy") else np.asarray(rec)
    err = float(np.max(np.abs(rec_np - x)))
    assert err < TOL_RECONSTRUCTION, f"reconstruction error {err} >= {TOL_RECONSTRUCTION}"


def test_stft_istft_reconstruction_odd_length() -> None:
    """Roundtrip must also hold for lengths not divisible by hop (edge case)."""
    rng = np.random.default_rng(123)
    x = (rng.standard_normal(1000).astype(np.float32)) * 0.4
    spec = compute_stft(x, n_fft=512, hop_length=128, window="hann")
    rec = compute_istft(spec, length=int(x.shape[0]), n_fft=512, hop_length=128)
    rec_np = rec.numpy() if hasattr(rec, "numpy") else np.asarray(rec)
    assert rec_np.shape == x.shape
    assert float(np.max(np.abs(rec_np - x))) < TOL_RECONSTRUCTION
