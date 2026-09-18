"""DSP preprocessing: resampling, pre-emphasis FIR, STFT/iSTFT (FR-2).

Audio contract is 16 kHz mono throughout (PRD FR-2). STFT defaults are
``N_fft=512``, ``hop_length=128``, Hanning window — shared with the
visualization and separator layers so analysis/synthesis stay consistent.
"""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import torch

#: Project-wide target sample rate (Hz).
TARGET_SAMPLE_RATE_HZ: int = 16000

#: Default pre-emphasis coefficient: y[n] = x[n] - coef * x[n-1].
PRE_EMPHASIS_COEF: float = 0.97

#: Default STFT parameters (FR-2 / course evidence: windowing + overlap-add).
N_FFT: int = 512
HOP_LENGTH: int = 128


def resample_audio(
    audio: np.ndarray | torch.Tensor,
    orig_sr: int,
    target_sr: int = TARGET_SAMPLE_RATE_HZ,
) -> np.ndarray | torch.Tensor:
    """Resample mono audio to ``target_sr`` (deterministic, identity-safe).

    Args:
        audio: 1-D mono waveform (NumPy or Torch).
        orig_sr: Input sample rate in Hz.
        target_sr: Output sample rate in Hz (default 16 kHz).

    Returns:
        Resampled 1-D ``float32`` waveform of the same kind as the input
        (NumPy in, NumPy out; Torch in, Torch out). Returned unchanged (as a
        ``float32`` copy) when ``orig_sr == target_sr``.

    Raises:
        ValueError: If audio is not 1-D or sample rates are not positive.
    """
    is_torch = isinstance(audio, torch.Tensor)
    arr = (
        audio.detach().cpu().float().reshape(-1).numpy()
        if is_torch
        else np.asarray(audio, dtype=np.float32)
    )
    if arr.ndim != 1:
        raise ValueError(f"resample_audio expects 1-D mono audio, got shape {arr.shape}.")
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError(f"Sample rates must be positive, got {orig_sr} -> {target_sr}.")
    if int(orig_sr) == int(target_sr):
        out = np.ascontiguousarray(arr, dtype=np.float32)
    else:
        out = np.ascontiguousarray(
            librosa.resample(arr, orig_sr=int(orig_sr), target_sr=int(target_sr)),
            dtype=np.float32,
        )
    if is_torch:
        return torch.from_numpy(out)
    return out


def to_mono(audio: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Downmix multi-channel audio to mono by channel averaging.

    Args:
        audio: 1-D (already mono) or 2-D ``[channels, time]`` / ``[time, channels]``.

    Returns:
        1-D mono ``float32`` waveform of the same kind as the input.
    """
    is_torch = isinstance(audio, torch.Tensor)
    arr = audio.detach().cpu().float().numpy() if is_torch else np.asarray(audio, dtype=np.float32)
    if arr.ndim == 1:
        out = np.ascontiguousarray(arr, dtype=np.float32)
        return torch.from_numpy(out) if is_torch else out
    if arr.ndim != 2:
        raise ValueError(f"to_mono expects 1-D or 2-D audio, got shape {arr.shape}.")
    # Heuristic: time axis is the longer one for speech clips.
    if arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
        mono = arr.mean(axis=0)
    elif arr.shape[1] <= 8 and arr.shape[0] > arr.shape[1]:
        mono = arr.mean(axis=1)
    else:  # ambiguous square input: average over axis 0.
        mono = arr.mean(axis=0)
    out = np.ascontiguousarray(mono, dtype=np.float32)
    if is_torch:
        return torch.from_numpy(out)
    return out


def ensure_16k_mono(
    audio: np.ndarray | torch.Tensor, orig_sr: int, target_sr: int = TARGET_SAMPLE_RATE_HZ
) -> np.ndarray | torch.Tensor:
    """Convert arbitrary channel layout + sample rate to 16 kHz mono PCM.

    Applies :func:`to_mono` then :func:`resample_audio` deterministically.

    Args:
        audio: 1-D mono or 2-D multi-channel waveform (NumPy or Torch).
        orig_sr: Input sample rate in Hz.
        target_sr: Output sample rate in Hz (default 16 kHz).

    Returns:
        1-D ``float32`` 16 kHz mono waveform of the same kind as the input.
    """
    mono = to_mono(audio)
    return resample_audio(mono, orig_sr=orig_sr, target_sr=target_sr)


def check_clipping(audio: np.ndarray | torch.Tensor, threshold: float = 0.999) -> bool:
    """Report whether a waveform exceeds a peak-amplitude threshold.

    Args:
        audio: 1-D waveform with nominal range ``[-1, 1]``.
        threshold: Absolute peak above which audio counts as clipped.

    Returns:
        ``True`` when ``max(abs(audio)) > threshold``.
    """
    arr = (
        audio.detach().cpu().float().numpy()
        if isinstance(audio, torch.Tensor)
        else np.asarray(audio, dtype=np.float32)
    )
    return bool(np.max(np.abs(arr)) > float(threshold))


def pre_emphasis(audio: np.ndarray, coef: float = PRE_EMPHASIS_COEF) -> np.ndarray:
    """Apply first-order high-frequency emphasis FIR filter.

    Implements ``y[n] = x[n] - coef * x[n-1]`` with ``y[0] = x[0]``
    (causal FIR with taps ``[1, -coef]``).

    Args:
        audio: 1-D mono waveform.
        coef: Feedback coefficient (default 0.97).

    Returns:
        Emphasized 1-D ``float32`` waveform, same length as input.
    """
    arr = np.ascontiguousarray(np.asarray(audio, dtype=np.float64))
    if arr.ndim != 1:
        raise ValueError(f"pre_emphasis expects 1-D audio, got shape {arr.shape}.")
    if arr.size == 0:
        return np.zeros(0, dtype=np.float32)
    out = np.empty_like(arr)
    out[0] = arr[0]
    out[1:] = arr[1:] - float(coef) * arr[:-1]
    return np.ascontiguousarray(out, dtype=np.float32)


def de_emphasis(audio: np.ndarray, coef: float = PRE_EMPHASIS_COEF) -> np.ndarray:
    """Invert :func:`pre_emphasis` via ``x[n] = y[n] + coef * x[n-1]``.

    Args:
        audio: 1-D emphasized waveform.
        coef: Coefficient used at emphasis time.

    Returns:
        Reconstructed 1-D ``float32`` waveform.
    """
    arr = np.ascontiguousarray(np.asarray(audio, dtype=np.float64))
    if arr.ndim != 1:
        raise ValueError(f"de_emphasis expects 1-D audio, got shape {arr.shape}.")
    out = np.empty_like(arr)
    if arr.size == 0:
        return np.zeros(0, dtype=np.float32)
    out[0] = arr[0]
    for n in range(1, arr.size):
        out[n] = arr[n] + float(coef) * out[n - 1]
    return np.ascontiguousarray(out, dtype=np.float32)


def pre_emphasis_frequency_response(
    freqs: np.ndarray, coef: float = PRE_EMPHASIS_COEF
) -> np.ndarray:
    """Closed-form FIR response ``H(e^{jw}) = 1 - coef * e^{-jw}``.

    Args:
        freqs: Normalized angular frequencies in radians/sample.
        coef: FIR coefficient.

    Returns:
        Complex frequency response sampled at ``freqs``.
    """
    w = np.asarray(freqs, dtype=np.float64)
    return (1.0 - float(coef) * np.exp(-1j * w)).astype(np.complex128)


def _hann_window(n_fft: int = N_FFT) -> torch.Tensor:
    """Return a periodic Hann window (matches ``torch.hann_window``)."""
    return torch.hann_window(int(n_fft))


def compute_stft(
    audio: np.ndarray | torch.Tensor,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    window: str = "hann",
) -> torch.Tensor:
    """Compute complex STFT with Hanning window and overlap-add support.

    Args:
        audio: 1-D mono waveform (NumPy or Torch).
        n_fft: FFT size (default 512).
        hop_length: Hop between frames (default 128).
        window: Only ``"hann"``/``"hanning"`` is supported (PRD frozen STFT).

    Returns:
        Complex spectrogram of shape ``[freq, time]`` where
        ``freq = n_fft // 2 + 1``.

    Raises:
        ValueError: For unsupported windows or non-1-D input.
    """
    if window.lower() not in ("hann", "hanning"):
        raise ValueError(f"Frozen STFT window must be Hann, got {window!r}.")
    if isinstance(audio, np.ndarray):
        wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    else:
        wav = torch.as_tensor(audio, dtype=torch.float32).reshape(-1)
    if wav.ndim != 1:
        raise ValueError(f"compute_stft expects 1-D audio, got shape {tuple(wav.shape)}.")
    return torch.stft(
        wav,
        n_fft=int(n_fft),
        hop_length=int(hop_length),
        window=_hann_window(int(n_fft)).to(wav.dtype),
        center=True,
        normalized=False,
        onesided=True,
        return_complex=True,
    )


def compute_istft(
    stft_matrix: torch.Tensor,
    length: int | None = None,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    window: str = "hann",
) -> torch.Tensor:
    """Invert :func:`compute_stft` via overlap-add iSTFT.

    Args:
        stft_matrix: Complex spectrogram of shape ``[freq, time]``.
        length: Optional target output length (trims/pads deterministically).
        n_fft: FFT size used at analysis time.
        hop_length: Hop used at analysis time.
        window: Must match the analysis window (Hann).

    Returns:
        1-D time-domain waveform.
    """
    if window.lower() not in ("hann", "hanning"):
        raise ValueError(f"Frozen iSTFT window must be Hann, got {window!r}.")
    spec = torch.as_tensor(stft_matrix)
    if not torch.is_complex(spec):
        raise ValueError("compute_istft expects a complex STFT matrix.")
    wav = torch.istft(
        spec,
        n_fft=int(n_fft),
        hop_length=int(hop_length),
        window=_hann_window(int(n_fft)).to(torch.float32),
        center=True,
        normalized=False,
        onesided=True,
        length=length,
    )
    return wav.reshape(-1)


def stft_roundtrip_error(
    audio: np.ndarray | torch.Tensor,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> float:
    """Measure max-absolute STFT/iSTFT reconstruction error.

    Args:
        audio: 1-D mono waveform.
        n_fft: FFT size. hop_length: Hop length.

    Returns:
        ``max(abs(istft(stft(x)) - x))`` as a Python float.
    """
    wav = (
        np.ascontiguousarray(np.asarray(audio), dtype=np.float32)
        if isinstance(audio, np.ndarray)
        else torch.as_tensor(audio, dtype=torch.float32).reshape(-1).numpy()
    )
    spec = compute_stft(wav, n_fft=n_fft, hop_length=hop_length)
    rec = compute_istft(spec, length=int(wav.shape[0]), n_fft=n_fft, hop_length=hop_length)
    return float(np.max(np.abs(rec.numpy() - wav)))


def ensure_parent(path: str | Path) -> Path:
    """Create a file's parent directory and return the resolved path."""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


__all__ = [
    "HOP_LENGTH",
    "N_FFT",
    "PRE_EMPHASIS_COEF",
    "TARGET_SAMPLE_RATE_HZ",
    "check_clipping",
    "compute_istft",
    "compute_stft",
    "de_emphasis",
    "ensure_16k_mono",
    "ensure_parent",
    "pre_emphasis",
    "pre_emphasis_frequency_response",
    "resample_audio",
    "stft_roundtrip_error",
    "to_mono",
]
