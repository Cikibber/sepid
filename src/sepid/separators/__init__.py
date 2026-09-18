"""SepID separator adapters (M0 mixture, M1 NMF, M2 neural, M3 oracle)."""
from __future__ import annotations

from sepid.separators.base import BaseSeparator, Separator
from sepid.separators.mixture import MixtureBaseline
from sepid.separators.oracle_mask import OracleMaskError, OracleMaskSeparator

__all__ = [
    "BaseSeparator",
    "MixtureBaseline",
    "OracleMaskError",
    "OracleMaskSeparator",
    "Separator",
]
