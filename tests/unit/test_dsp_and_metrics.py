"""Exit-gate unit tests: STFT identity, SI-SDR on sinusoids, M3 dual-tone gain.

Covers PRD §8.4 / FR-2 / FR-3 / FR-6 / FR-7:

1. STFT (N_fft=512, hop=128, Hann) / iSTFT round-trip identity (< 1e-4).
2. SI-SDR against deterministic synthetic sinusoids (identity ~ high,
   permutation selection correct, SDR finite).
3. M3 oracle (IRM + IBM) achieves > 10 dB mean SI-SDRi on a synthetic
   dual-tone mixture; M3 without references raises.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sepid.audio.preprocessing import (  # noqa: E402
    HOP_LENGTH,
    N_FFT,
    TARGET_SAMPLE_RATE_HZ,
    compute_istft,
    compute_stft,
    ensure_16k_mono,
    pre_emphasis,
    resample_audio,
    to_mono,
)
from sepid.audio.visualization import figure_path, plot_spectrogram, plot_waveform  # noqa: E402
from sepid.evaluation.signal import evaluate_mixture, permutation_si_sdr, sdr, si_sdr  # noqa: E402
from sepid.separators.base import BaseSeparator  # noqa: E402
from sepid.separators.mixture import MixtureBaseline  # noqa: E402
from sepid.separators.oracle_mask import OracleMaskError, OracleMaskSeparator  # noqa: E402

SR = TARGET_SAMPLE_RATE_HZ
TOL_RECONSTRUCTION = 1e-4


def _tone(freq_hz: float, seconds: float = 1.0, amplitude: float = 0.5) -> np.ndarray:
    """Deterministic sine tone at 16 kHz."""
    t = np.arange(int(SR * seconds), dtype=np.float64) / float(SR)
    return (amplitude * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)


def test_stft_roundtrip_identity_sine_and_noise() -> None:
    """STFT/iSTFT round-trip max-abs error is < 1e-4."""
    assert N_FFT == 512
    assert HOP_LENGTH == 128
    rng = np.random.default_rng(0)
    x = (_tone(440.0) + 0.2 * _tone(880.0)).astype(np.float32)
    x = (x + 0.01 * rng.standard_normal(x.shape).astype(np.float32)).astype(np.float32)
    spec = compute_stft(x, n_fft=512, hop_length=128, window="hann")
    assert spec.shape[0] == 257
    rec = compute_istft(spec, length=x.shape[0], n_fft=512, hop_length=128)
    err = float(np.max(np.abs(rec.numpy() - x)))
    assert err < TOL_RECONSTRUCTION, f"round-trip error {err}"


def test_resample_to_mono_16k_contract() -> None:
    """8 kHz stereo input converts deterministically to 16 kHz mono."""
    t8 = np.arange(8000, dtype=np.float32) / 8000.0
    left = np.sin(2.0 * np.pi * 440.0 * t8).astype(np.float32)
    stereo = np.stack([left, left], axis=0)
    mono16 = ensure_16k_mono(stereo, orig_sr=8000)
    assert isinstance(mono16, np.ndarray)
    assert mono16.shape == (16000,)
    assert mono16.dtype == np.float32
    # Torch path preserves kind.
    torch_out = ensure_16k_mono(torch.from_numpy(stereo), orig_sr=8000)
    assert isinstance(torch_out, torch.Tensor)
    assert tuple(torch_out.shape) == (16000,)
    # Identity path is exact.
    same = resample_audio(_tone(440.0, seconds=0.1), orig_sr=16000)
    assert same.shape == (1600,)
    assert to_mono(np.ones((2, 100), dtype=np.float32)).shape == (100,)


def test_pre_emphasis_fir_equation() -> None:
    """Pre-emphasis matches y[n] = x[n] - 0.97 * x[n-1]."""
    x = np.array([1.0, 2.0, 3.0, 0.5], dtype=np.float32)
    y = pre_emphasis(x, coef=0.97)
    expected = np.array([1.0, 2.0 - 0.97, 3.0 - 0.97 * 2.0, 0.5 - 0.97 * 3.0])
    np.testing.assert_allclose(y, expected, rtol=1e-6, atol=1e-7)


def test_si_sdr_identity_high_and_scaled_invariant() -> None:
    """Identical and gain-scaled estimates score high; noise scores low."""
    ref = _tone(440.0)
    assert si_sdr(ref, ref.copy()) > 60.0
    assert si_sdr(ref, 2.5 * ref) > 60.0  # scale invariance
    rng = np.random.default_rng(1)
    noisy = (ref + 0.5 * rng.standard_normal(ref.shape)).astype(np.float32)
    assert si_sdr(ref, noisy) < si_sdr(ref, ref.copy())
    assert np.isfinite(sdr(ref, noisy))


def test_permutation_selection_picks_max_mean() -> None:
    """Swapped estimates select P2 (index 1) with the same max mean."""
    s1, s2 = _tone(440.0), _tone(660.0)
    refs = np.stack([s1, s2])
    est_identity = np.stack([s1, s2])
    est_swapped = np.stack([s2, s1])
    r1 = permutation_si_sdr(est_identity, refs)
    r2 = permutation_si_sdr(est_swapped, refs)
    assert r1["permutation"] == 0
    assert r1["permutation_tuple"] == [0, 1]
    assert r2["permutation"] == 1
    assert r2["permutation_tuple"] == [1, 0]
    assert r2["mean_si_sdr"] == pytest.approx(r1["mean_si_sdr"], abs=1e-6)


def test_evaluate_mixture_reports_improvement_and_permutation() -> None:
    """evaluate_mixture returns SI-SDRi/SDRi plus the selected permutation."""
    s1, s2 = _tone(440.0), _tone(660.0)
    mix = (s1 + s2).astype(np.float32)
    refs = np.stack([s1, s2])
    oracle = OracleMaskSeparator(mask="soft")
    est = oracle.separate(torch.from_numpy(mix), torch.from_numpy(refs)).numpy()
    result = evaluate_mixture(est, refs, mix)
    for key in (
        "permutation",
        "permutation_tuple",
        "si_sdr",
        "sdr",
        "mean_si_sdr",
        "input_si_sdr",
        "si_sdri",
        "sdri",
        "mean_si_sdri",
        "mean_sdri",
    ):
        assert key in result, f"missing metric key: {key}"
    assert result["permutation"] in (0, 1)
    assert len(result["si_sdri"]) == 2
    assert result["mean_si_sdri"] > 10.0


def test_m0_duplicates_mixture_shapes() -> None:
    """M0 accepts [time] and [1, time], returns [2, time], input untouched."""
    m0 = MixtureBaseline()
    assert isinstance(m0, BaseSeparator)
    mix = torch.from_numpy(_tone(440.0, seconds=0.25))
    snapshot = mix.clone()
    out = m0.separate(mix)
    assert tuple(out.shape) == (2, mix.numel())
    torch.testing.assert_close(out[0], snapshot)
    torch.testing.assert_close(out[1], snapshot)
    torch.testing.assert_close(mix, snapshot)  # no in-place change
    out_batched = m0.separate(mix.reshape(1, -1))
    assert tuple(out_batched.shape) == (2, mix.numel())
    with pytest.raises(ValueError):
        m0.separate(torch.zeros(2, 100))


def test_m3_oracle_masks_gain_over_10db_and_requires_refs() -> None:
    """M3 IRM/IBM each gain > 10 dB SI-SDRi; missing refs raise."""
    s1 = _tone(440.0, amplitude=0.5)
    s2 = _tone(440.0 * 1.5, amplitude=0.5)
    mix = (s1 + s2).astype(np.float32)
    refs = np.stack([s1, s2])
    for mask in ("soft", "binary"):
        oracle = OracleMaskSeparator(mask=mask)
        est = oracle.separate(torch.from_numpy(mix), torch.from_numpy(refs)).numpy()
        assert est.shape == (2, mix.shape[0])
        result = evaluate_mixture(est, refs, mix)
        assert result["mean_si_sdri"] > 10.0, f"{mask}: {result['mean_si_sdri']}"
    oracle = OracleMaskSeparator()
    with pytest.raises(OracleMaskError):
        oracle.separate(torch.from_numpy(mix))
    with pytest.raises(OracleMaskError):
        oracle.separate(torch.from_numpy(mix), torch.zeros(2, mix.shape[0] - 1))
    with pytest.raises(ValueError):
        OracleMaskSeparator(mask="wiener")


def test_visualization_saves_run_bundle_figures(tmp_path: Path) -> None:
    """Waveform + spectrogram save under results/<run_id>/figures/ with units."""
    monkey_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        x = _tone(440.0, seconds=0.5)
        wave = figure_path("ut-run", "waveform.png")
        spec = figure_path("ut-run", "spectrogram.png")
        assert str(wave) == str(Path("results") / "ut-run" / "figures" / "waveform.png")
        out_w = plot_waveform(x, run_id="ut-run", filename="waveform.png")
        out_s = plot_spectrogram(x, run_id="ut-run", filename="spectrogram.png")
        assert out_w.is_file() and out_s.is_file()
        assert out_w.stat().st_size > 0 and out_s.stat().st_size > 0
    finally:
        os.chdir(monkey_cwd)
