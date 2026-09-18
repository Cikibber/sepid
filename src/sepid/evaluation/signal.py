"""Signal metrics: permutation-aware SI-SDR/SDR + improvements (PRD §14.1, FR-7).

For each mixture both source permutations are evaluated::

    P1: estimate[0] -> reference[0], estimate[1] -> reference[1]
    P2: estimate[0] -> reference[1], estimate[1] -> reference[0]

The permutation maximizing mean SI-SDR wins; the same stored assignment is
used for linked per-source SDR analysis. Improvement metrics are
``output - input`` where the input is the unprocessed mixture scored against
each reference under the winning assignment.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _as_1d(label: str, x: np.ndarray) -> np.ndarray:
    """Validate a 1-D waveform operand.

    Args:
        label: Operand name for error messages.
        x: Candidate waveform.

    Returns:
        1-D ``float64`` array.

    Raises:
        ValueError: For non-1-D or empty input.
    """
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{label} must be non-empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} contains NaN or Inf.")
    return arr


def si_sdr(reference: np.ndarray, estimate: np.ndarray, eps: float = 1e-8) -> float:
    """Scale-invariant signal-to-distortion ratio.

    Implements the standard projection formulation: the estimate is scored
    against the optimally scaled reference, so pure gain differences do not
    penalize the score.

    Args:
        reference: 1-D clean reference waveform.
        estimate: 1-D estimated waveform (same length).
        eps: Floor for denominators and silence handling.

    Returns:
        SI-SDR in dB.

    Raises:
        ValueError: For length mismatch, empty input, or non-finite values.
    """
    ref = _as_1d("reference", reference)
    est = _as_1d("estimate", estimate)
    if ref.shape != est.shape:
        raise ValueError(
            f"reference and estimate lengths differ: {ref.shape} vs {est.shape}."
        )
    ref_energy = float(np.dot(ref, ref)) + float(eps)
    scale = float(np.dot(est, ref)) / ref_energy
    target = scale * ref
    noise = est - target
    ratio = float(np.dot(target, target)) / (float(np.dot(noise, noise)) + float(eps))
    return float(10.0 * np.log10(max(ratio, float(eps))))


def sdr(reference: np.ndarray, estimate: np.ndarray, eps: float = 1e-8) -> float:
    """Classical (non-scale-invariant) signal-to-distortion ratio.

    Args:
        reference: 1-D clean reference waveform.
        estimate: 1-D estimated waveform (same length).
        eps: Floor for denominators and silence handling.

    Returns:
        SDR in dB.

    Raises:
        ValueError: For length mismatch, empty input, or non-finite values.
    """
    ref = _as_1d("reference", reference)
    est = _as_1d("estimate", estimate)
    if ref.shape != est.shape:
        raise ValueError(
            f"reference and estimate lengths differ: {ref.shape} vs {est.shape}."
        )
    noise = est - ref
    ratio = float(np.dot(ref, ref)) / (float(np.dot(noise, noise)) + float(eps))
    return float(10.0 * np.log10(max(ratio, float(eps))))


def _as_2xT(label: str, x: np.ndarray) -> np.ndarray:
    """Validate a ``[2, time]`` metric operand.

    Args:
        label: Operand name for error messages.
        x: Candidate ``[2, time]`` array.

    Returns:
        ``float64`` array of shape ``[2, time]``.

    Raises:
        ValueError: For wrong shape, empty, or non-finite input.
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != 2 or arr.shape[1] == 0:
        raise ValueError(f"{label} must have shape [2, time], got {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} contains NaN or Inf.")
    return arr


def permutation_si_sdr(
    estimates: np.ndarray, references: np.ndarray, eps: float = 1e-8
) -> dict[str, Any]:
    """Score both permutations and pick the max-mean-SI-SDR assignment.

    Args:
        estimates: Array of shape ``[2, time]``.
        references: Array of shape ``[2, time]``.
        eps: Metric epsilon policy (stored in the result).

    Returns:
        Dict with ``permutation`` (0 = P1 identity, 1 = P2 swap),
        ``permutation_tuple`` (index mapping), per-source ``si_sdr``
        and ``sdr`` lists under the winning assignment, ``mean_si_sdr``,
        ``mean_sdr``, and ``eps``.

    Raises:
        ValueError: For shape mismatch or invalid input.
    """
    est = _as_2xT("estimates", estimates)
    ref = _as_2xT("references", references)
    if est.shape[1] != ref.shape[1]:
        raise ValueError(
            f"estimates and references lengths differ: {est.shape} vs {ref.shape}."
        )
    perms: tuple[tuple[int, int], ...] = ((0, 1), (1, 0))
    scored: list[tuple[float, int]] = []
    per_source_sisdr: list[list[float]] = []
    per_source_sdr: list[list[float]] = []
    for index, perm in enumerate(perms):
        s = [si_sdr(ref[perm[k]], est[k], eps=eps) for k in (0, 1)]
        d = [sdr(ref[perm[k]], est[k], eps=eps) for k in (0, 1)]
        per_source_sisdr.append(s)
        per_source_sdr.append(d)
        scored.append((float(np.mean(s)), index))
    best = max(scored)[1]
    return {
        "permutation": int(best),
        "permutation_tuple": [int(v) for v in perms[best]],
        "si_sdr": [float(v) for v in per_source_sisdr[best]],
        "sdr": [float(v) for v in per_source_sdr[best]],
        "mean_si_sdr": float(np.mean(per_source_sisdr[best])),
        "mean_sdr": float(np.mean(per_source_sdr[best])),
        "eps": float(eps),
    }


def evaluate_mixture(
    estimates: np.ndarray,
    references: np.ndarray,
    mixture: np.ndarray,
    eps: float = 1e-8,
) -> dict[str, Any]:
    """Full per-mixture scoring: output metrics plus SI-SDRi/SDRi.

    Args:
        estimates: Separated output of shape ``[2, time]``.
        references: Clean references of shape ``[2, time]``.
        mixture: Unprocessed mixture of shape ``[time]``.
        eps: Metric epsilon policy.

    Returns:
        Dict with all :func:`permutation_si_sdr` fields plus ``input_si_sdr``
        (mixture vs each reference under the winning assignment),
        ``input_sdr``, ``si_sdri`` / ``sdr`` per-source improvements,
        ``mean_si_sdri``, and ``mean_sdri``.
    """
    result = permutation_si_sdr(estimates, references, eps=eps)
    mix = _as_1d("mixture", mixture)
    ref = _as_2xT("references", references)
    if mix.shape[0] != ref.shape[1]:
        raise ValueError(
            f"mixture length {mix.shape[0]} differs from references length {ref.shape[1]}."
        )
    perm: list[int] = [int(v) for v in result["permutation_tuple"]]
    input_sisdr = [si_sdr(ref[perm[k]], mix, eps=eps) for k in (0, 1)]
    input_sdr = [sdr(ref[perm[k]], mix, eps=eps) for k in (0, 1)]
    si_sdri = [float(o) - float(i) for o, i in zip(result["si_sdr"], input_sisdr)]
    sdri = [float(o) - float(i) for o, i in zip(result["sdr"], input_sdr)]
    result.update(
        {
            "input_si_sdr": [float(v) for v in input_sisdr],
            "input_sdr": [float(v) for v in input_sdr],
            "si_sdri": si_sdri,
            "sdri": sdri,
            "mean_si_sdri": float(np.mean(si_sdri)),
            "mean_sdri": float(np.mean(sdri)),
        }
    )
    return result


__all__ = ["evaluate_mixture", "permutation_si_sdr", "sdr", "si_sdr"]
