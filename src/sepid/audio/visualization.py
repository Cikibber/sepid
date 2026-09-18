"""Aligned waveform / spectrogram figures (FR-3).

Every figure carries units, sample rate, color scale, and run ID (FR-3).
Outputs are written deterministically to ``results/figures/``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from sepid.audio.preprocessing import N_FFT, HOP_LENGTH, TARGET_SAMPLE_RATE_HZ, compute_stft, ensure_parent

#: Default figure output directory (PRD §11.4 run bundle: ``figures/``).
DEFAULT_FIGURE_DIR: Path = Path("results/figures")


def figure_path(run_id: str, filename: str, results_root: str | Path = "results") -> Path:
    """Resolve a deterministic figure path ``results/<run_id>/figures/<filename>``.

    Args:
        run_id: Run identifier (PRD §11.4 bundle directory).
        filename: Figure file name (e.g. ``"waveform.png"``).
        results_root: Root results directory (default ``"results"``).

    Returns:
        Path with parent directories created.
    """
    out = Path(results_root) / str(run_id) / "figures" / str(filename)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _resolve_output(
    output_path: str | Path | None, run_id: str, default_name: str
) -> Path:
    """Use explicit ``output_path`` or fall back to ``results/<run_id>/figures/``."""
    if output_path is not None:
        return ensure_parent(output_path)
    return figure_path(run_id, default_name)


def _to_mono_float(audio: np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert waveform input to 1-D float32 NumPy."""
    if isinstance(audio, torch.Tensor):
        arr = audio.detach().cpu().float().reshape(-1).numpy()
    else:
        arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    return np.ascontiguousarray(arr, dtype=np.float32)


def magnitude_spectrogram_db(
    audio: np.ndarray | torch.Tensor,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    floor_db: float = -80.0,
) -> np.ndarray:
    """Compute log-magnitude (dB) spectrogram for visualization.

    Args:
        audio: 1-D mono waveform.
        n_fft: FFT size. hop_length: Hop length.
        floor_db: Dynamic-range floor in dB.

    Returns:
        2-D ``[freq, time]`` array in dB, peak-normalized to 0 dB.
    """
    spec = compute_stft(audio, n_fft=n_fft, hop_length=hop_length)
    mag = torch.abs(spec).numpy().astype(np.float64)
    peak = float(mag.max()) if mag.size else 1.0
    mag = mag / max(peak, 1e-12)
    return (20.0 * np.log10(np.maximum(mag, 10.0 ** (float(floor_db) / 20.0)))).astype(np.float32)


def plot_waveform(
    audio: np.ndarray | torch.Tensor,
    output_path: str | Path | None = None,
    sample_rate_hz: int = TARGET_SAMPLE_RATE_HZ,
    title: str = "Waveform",
    run_id: str = "unregistered",
    results_root: str | Path = "results",
    filename: str = "waveform.png",
) -> Path:
    """Plot a time-domain waveform with time (s) and amplitude units.

    Args:
        audio: 1-D mono waveform.
        output_path: Destination PNG path. When ``None``, saves to
            ``results/<run_id>/figures/<filename>``.
        sample_rate_hz: Sample rate annotation.
        title: Panel title. run_id: Run identifier stamped on the figure.
        results_root: Root results directory.
        filename: Default file name used with ``run_id`` routing.

    Returns:
        Resolved output path.
    """
    wav = _to_mono_float(audio)
    times = np.arange(wav.shape[0], dtype=np.float64) / float(sample_rate_hz)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(times, wav, linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"{title} — {sample_rate_hz} Hz (run: {run_id})")
    ax.set_xlim(0.0, max(times[-1] if times.size else 0.0, 1e-9))
    fig.tight_layout()
    out = _resolve_output(output_path, run_id, filename)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_spectrogram(
    audio: np.ndarray | torch.Tensor,
    output_path: str | Path | None = None,
    sample_rate_hz: int = TARGET_SAMPLE_RATE_HZ,
    title: str = "Spectrogram",
    run_id: str = "unregistered",
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    results_root: str | Path = "results",
    filename: str = "spectrogram.png",
) -> Path:
    """Plot a magnitude spectrogram with time/frequency units and dB scale.

    Args:
        audio: 1-D mono waveform.
        output_path: Destination PNG path. When ``None``, saves to
            ``results/<run_id>/figures/<filename>``.
        sample_rate_hz: Sample rate annotation.
        title: Panel title. run_id: Run identifier stamped on the figure.
        n_fft: FFT size annotation. hop_length: Hop annotation.
        results_root: Root results directory.
        filename: Default file name used with ``run_id`` routing.

    Returns:
        Resolved output path.
    """
    spec_db = magnitude_spectrogram_db(audio, n_fft=n_fft, hop_length=hop_length)
    extent = [0.0, spec_db.shape[1] * hop_length / float(sample_rate_hz), 0.0, sample_rate_hz / 2.0]
    fig, ax = plt.subplots(figsize=(10, 3.5))
    im = ax.imshow(spec_db, origin="lower", aspect="auto", extent=extent, cmap="magma")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"{title} — N_fft={n_fft}, hop={hop_length}, Hann (run: {run_id})")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Magnitude (dB, peak = 0 dB)")
    fig.tight_layout()
    out = _resolve_output(output_path, run_id, filename)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_before_after(
    before: np.ndarray | torch.Tensor,
    after: np.ndarray | torch.Tensor,
    output_path: str | Path | None = None,
    labels: Sequence[str] = ("mixture", "estimate"),
    sample_rate_hz: int = TARGET_SAMPLE_RATE_HZ,
    title: str = "Before / after separation",
    run_id: str = "unregistered",
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    results_root: str | Path = "results",
    filename: str = "before_after.png",
) -> Path:
    """Plot aligned before/after waveform + spectrogram panels (FR-3).

    Args:
        before: Input mixture waveform. after: Separated estimate waveform.
        output_path: Destination PNG path. When ``None``, saves to
            ``results/<run_id>/figures/<filename>``.
        labels: Panel labels ``(before, after)``.
        sample_rate_hz: Sample rate annotation.
        title: Figure title. run_id: Run identifier stamped on the figure.
        n_fft: FFT size. hop_length: Hop length.
        results_root: Root results directory.
        filename: Default file name used with ``run_id`` routing.

    Returns:
        Resolved output path.
    """
    wav_before = _to_mono_float(before)
    wav_after = _to_mono_float(after)
    spec_before = magnitude_spectrogram_db(wav_before, n_fft=n_fft, hop_length=hop_length)
    spec_after = magnitude_spectrogram_db(wav_after, n_fft=n_fft, hop_length=hop_length)
    times_b = np.arange(wav_before.shape[0]) / float(sample_rate_hz)
    times_a = np.arange(wav_after.shape[0]) / float(sample_rate_hz)

    fig, axes = plt.subplots(2, 2, figsize=(12, 6), constrained_layout=True)
    fig.suptitle(f"{title} — {sample_rate_hz} Hz (run: {run_id})")
    for ax, times, wav, label in zip(
        (axes[0, 0], axes[0, 1]), (times_b, times_a), (wav_before, wav_after), labels
    ):
        ax.plot(times, wav, linewidth=0.7)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(label)
    for ax, spec, label in zip(
        (axes[1, 0], axes[1, 1]), (spec_before, spec_after), labels
    ):
        extent = [0.0, spec.shape[1] * hop_length / float(sample_rate_hz), 0.0, sample_rate_hz / 2.0]
        im = ax.imshow(spec, origin="lower", aspect="auto", extent=extent, cmap="magma")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(f"{label} magnitude (dB)")
        fig.colorbar(im, ax=ax, label="dB (peak = 0 dB)")
    out = _resolve_output(output_path, run_id, filename)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


__all__ = [
    "DEFAULT_FIGURE_DIR",
    "figure_path",
    "magnitude_spectrogram_db",
    "plot_before_after",
    "plot_spectrogram",
    "plot_waveform",
]
