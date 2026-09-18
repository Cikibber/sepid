"""SepID separator adapters (M0 mixture, M1 NMF, M2 neural, M3 oracle)."""
from __future__ import annotations

from sepid.separators.base import BaseSeparator, Separator
from sepid.separators.mixture import MixtureBaseline
from sepid.separators.neural import CHECKPOINT_ID, CHECKPOINT_REVISION, NeuralSeparator, NeuralSeparatorError
from sepid.separators.nmf_enrolled import EnrolledNMFSeparator
from sepid.separators.oracle_mask import OracleMaskError, OracleMaskSeparator

__all__ = [
    "BaseSeparator",
    "CHECKPOINT_ID",
    "CHECKPOINT_REVISION",
    "EnrolledNMFSeparator",
    "MixtureBaseline",
    "NeuralSeparator",
    "NeuralSeparatorError",
    "OracleMaskError",
    "OracleMaskSeparator",
    "Separator",
]
