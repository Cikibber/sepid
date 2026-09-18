"""Permutation-aware source assignment (PRD §14.1).

Selects the permutation maximizing mean SI-SDR and reuses that stored
assignment for linked per-source analyses (see :mod:`sepid.evaluation.signal`).
"""
from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from sepid.evaluation.signal import permutation_si_sdr


def best_permutation_by_sisdr(
    estimates: np.ndarray, references: np.ndarray, eps: float = 1e-8
) -> tuple[tuple[int, ...], float]:
    """Select the source permutation maximizing mean SI-SDR.

    Args:
        estimates: Array of shape ``[2, time]``.
        references: Array of shape ``[2, time]``.
        eps: Metric epsilon policy.

    Returns:
        Tuple of (best permutation index mapping, mean SI-SDR).
    """
    result: dict[str, Any] = permutation_si_sdr(estimates, references, eps=eps)
    return tuple(int(v) for v in result["permutation_tuple"]), float(result["mean_si_sdr"])


def all_permutation_scores(
    estimates: np.ndarray, references: np.ndarray, eps: float = 1e-8
) -> list[dict[str, Any]]:
    """Score every source permutation (diagnostics, PRD §14.1 step 2).

    Args:
        estimates: Array of shape ``[2, time]``.
        references: Array of shape ``[2, time]``.
        eps: Metric epsilon policy.

    Returns:
        One record per permutation with index mapping and mean SI-SDR.
    """
    from sepid.evaluation.signal import si_sdr

    est = np.asarray(estimates, dtype=np.float64)
    ref = np.asarray(references, dtype=np.float64)
    records: list[dict[str, Any]] = []
    for perm in itertools.permutations((0, 1)):
        scores = [si_sdr(ref[perm[k]], est[k], eps=eps) for k in (0, 1)]
        records.append(
            {
                "permutation_tuple": [int(v) for v in perm],
                "si_sdr": [float(v) for v in scores],
                "mean_si_sdr": float(np.mean(scores)),
            }
        )
    return records


__all__ = ["all_permutation_scores", "best_permutation_by_sisdr"]
