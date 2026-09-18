"""M1 enrolled semi-supervised NMF separator (PRD §8.2, FR-4).

Pipeline: magnitude STFTs of enrollment clips → KL-divergence NMF extracts
``W1``/``W2`` (K bases each) → frozen concatenated dictionary ``W=[W1|W2]``
→ fit activations ``H`` on the mixture magnitude STFT (W fixed) →
soft Wiener time-frequency masking → iSTFT. Enrollment utterances must be
recording- and utterance-disjoint from evaluation mixtures (PRD §9.5).
Frozen STFT: ``N_fft=512``, ``hop_length=128``, Hann window.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.decomposition import NMF

from sepid.audio.preprocessing import HOP_LENGTH, N_FFT, compute_istft, compute_stft
from sepid.separators.base import BaseSeparator, _as_mono_1d


def _magnitude_stft(x: np.ndarray, n_fft: int, hop_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute complex STFT and non-negative magnitude (floored at eps).

    Args:
        x: 1-D mono waveform.
        n_fft: FFT size. hop_length: Hop length.

    Returns:
        Tuple of (complex spectrogram, magnitude spectrogram).
    """
    spec = compute_stft(x, n_fft=n_fft, hop_length=hop_length)
    mag = torch.abs(spec).clamp_min(1e-10)
    return spec, mag


def _fit_dictionary(mag: torch.Tensor, n_components: int, n_iter: int, seed: int) -> np.ndarray:
    """Fit a KL-divergence NMF dictionary on a magnitude spectrogram.

    Args:
        mag: Magnitude spectrogram of shape ``[freq, time]``.
        n_components: Number of bases (K per speaker).
        n_iter: Maximum NMF iterations.
        seed: Deterministic random state.

    Returns:
        Dictionary matrix ``W`` of shape ``[freq, n_components]``.

    Raises:
        ValueError: If the spectrogram has fewer frames than components.
    """
    v = mag.numpy().astype(np.float64)
    if v.shape[1] < int(n_components):
        raise ValueError(
            f"Enrollment clip too short for K={n_components}: "
            f"only {v.shape[1]} STFT frames."
        )
    # NOTE: only the 'mu' solver handles KL divergence in sklearn.
    model = NMF(
        n_components=int(n_components),
        init="nndsvda",
        solver="mu",
        beta_loss="kullback-leibler",
        max_iter=int(n_iter),
        random_state=int(seed),
        tol=1e-4,
    )
    w = np.asarray(model.fit_transform(v), dtype=np.float64)  # [freq, K].
    return np.maximum(w, 1e-10)


def _fit_activations(
    mag: torch.Tensor, dictionary: np.ndarray, n_iter: int, seed: int
) -> np.ndarray:
    """Fit activations ``H`` on the mixture with the dictionary frozen.

    Uses KL-divergence multiplicative updates on ``H`` only
    (``W`` stays fixed)::

        H <- H * (W^T (V / (W H))) / (W^T 1)

    Args:
        mag: Mixture magnitude spectrogram of shape ``[freq, time]``.
        dictionary: Frozen ``W=[W1|W2]`` of shape ``[freq, 2K]``.
        n_iter: Number of MU iterations.
        seed: Deterministic random state.

    Returns:
        Activation matrix ``H`` of shape ``[2K, time]``.
    """
    v = mag.numpy().astype(np.float64)
    rng = np.random.default_rng(int(seed))
    w = np.maximum(np.asarray(dictionary, dtype=np.float64), 1e-10)
    h = np.maximum(rng.random((w.shape[1], v.shape[1])), 1e-10)
    ones = np.ones_like(v)
    eps = 1e-10
    for _ in range(int(n_iter)):
        wh = np.maximum(w @ h, eps)
        h *= (w.T @ (v / wh)) / (w.T @ ones + eps)
        h = np.maximum(h, eps)
    return np.ascontiguousarray(h)


def separate(
    mixture: np.ndarray,
    enrollment_s1: np.ndarray,
    enrollment_s2: np.ndarray,
    n_components: int = 32,
    n_iter: int = 100,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    seed: int = 0,
) -> np.ndarray:
    """Separate a mixture with speaker-enrolled semi-supervised NMF.

    Args:
        mixture: 1-D mono mixture waveform.
        enrollment_s1: Clean enrollment clip for speaker 1 (disjoint from
            mixture sources per PRD §9.5).
        enrollment_s2: Clean enrollment clip for speaker 2.
        n_components: Bases per speaker (K=32 default).
        n_iter: NMF iterations for dictionary + activation fits.
        n_fft: Frozen FFT size (default 512).
        hop_length: Frozen hop (default 128).
        seed: Deterministic random state.

    Returns:
        Array of shape ``[2, time]`` with separated waveforms at the
        exact input length.

    Raises:
        ValueError: For empty/non-1-D input or short enrollment clips.
    """
    mix = np.ascontiguousarray(np.asarray(mixture, dtype=np.float32)).reshape(-1)
    e1 = np.ascontiguousarray(np.asarray(enrollment_s1, dtype=np.float32)).reshape(-1)
    e2 = np.ascontiguousarray(np.asarray(enrollment_s2, dtype=np.float32)).reshape(-1)
    if mix.size == 0 or e1.size == 0 or e2.size == 0:
        raise ValueError("M1 inputs (mixture, enrollment_s1, enrollment_s2) must be non-empty.")
    mix_spec, mix_mag = _magnitude_stft(mix, n_fft, hop_length)
    _, e1_mag = _magnitude_stft(e1, n_fft, hop_length)
    _, e2_mag = _magnitude_stft(e2, n_fft, hop_length)
    w1 = _fit_dictionary(e1_mag, n_components, n_iter, seed)
    w2 = _fit_dictionary(e2_mag, n_components, n_iter, seed + 1)
    dictionary = np.concatenate([w1, w2], axis=1)  # W = [W1 | W2], frozen.
    h = _fit_activations(mix_mag, dictionary, n_iter, seed + 2)
    k = int(n_components)
    v1 = dictionary[:, :k] @ h[:k, :]
    v2 = dictionary[:, k:] @ h[k:, :]
    v_sum = (v1 + v2).clip(min=1e-10)
    mask1 = v1 / v_sum  # soft Wiener-style ratio mask.
    mask2 = 1.0 - mask1
    dtype = mix_spec.dtype
    est1 = compute_istft(
        mix_spec * torch.from_numpy(mask1).to(dtype),
        length=mix.shape[0],
        n_fft=n_fft,
        hop_length=hop_length,
    )
    est2 = compute_istft(
        mix_spec * torch.from_numpy(mask2).to(dtype),
        length=mix.shape[0],
        n_fft=n_fft,
        hop_length=hop_length,
    )
    return np.stack([est1.numpy(), est2.numpy()], axis=0).astype(np.float32)


class EnrolledNMFSeparator(BaseSeparator):
    """Speaker-enrolled semi-supervised NMF separator (PRD §8.1 M1).

    Attributes:
        name: Always ``"nmf_enrolled"``.
        sample_rate_hz: Native sample rate (16 kHz contract).
        num_sources: Always 2 in v1.
        n_components: Bases per speaker dictionary.
        n_iter: NMF iterations. seed: Deterministic random state.
    """

    name: str = "nmf_enrolled"
    sample_rate_hz: int = 16000
    num_sources: int = 2

    def __init__(self, n_components: int = 32, n_iter: int = 100, seed: int = 0) -> None:
        """Create the separator with frozen hyperparameters.

        Args:
            n_components: Bases per speaker (K=32 default).
            n_iter: NMF iterations (default 100).
            seed: Deterministic random state.
        """
        self.n_components: int = int(n_components)
        self.n_iter: int = int(n_iter)
        self.seed: int = int(seed)

    def separate(
        self,
        mixture: torch.Tensor,
        enrollment_s1: torch.Tensor | np.ndarray | None = None,
        enrollment_s2: torch.Tensor | np.ndarray | None = None,
    ) -> torch.Tensor:
        """Separate using frozen speaker dictionaries from enrollment clips.

        Args:
            mixture: Tensor of shape ``[time]`` or ``[1, time]``.
            enrollment_s1: Clean enrollment clip for speaker 1. Required.
            enrollment_s2: Clean enrollment clip for speaker 2. Required.

        Returns:
            Tensor of shape ``[2, time]`` at the exact input length.

        Raises:
            ValueError: If enrollment clips are missing (M1 is
                speaker-dependent and cannot run enrollment-free).
        """
        wav = _as_mono_1d(mixture)
        if enrollment_s1 is None or enrollment_s2 is None:
            raise ValueError(
                "M1 enrolled NMF requires clean enrollment clips for both speakers; "
                "it cannot separate unknown speakers enrollment-free."
            )
        e1 = torch.as_tensor(np.asarray(enrollment_s1), dtype=torch.float32).reshape(-1).numpy()
        e2 = torch.as_tensor(np.asarray(enrollment_s2), dtype=torch.float32).reshape(-1).numpy()
        out = separate(
            wav.numpy(),
            e1,
            e2,
            n_components=self.n_components,
            n_iter=self.n_iter,
            seed=self.seed,
        )
        return self._check_output(torch.from_numpy(out), wav.numel())


__all__ = ["EnrolledNMFSeparator", "separate"]
