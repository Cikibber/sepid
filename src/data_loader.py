"""Compatibility shim: ``src/data_loader.py`` re-exports the canonical loader.

Canonical implementation: :mod:`sepid.data.loader` (PRD §11).
"""
from __future__ import annotations

from sepid.data.loader import DatasetConfig, SepIDDataset, TARGET_SAMPLE_RATE_HZ, load_dataset_config

__all__ = ["DatasetConfig", "SepIDDataset", "TARGET_SAMPLE_RATE_HZ", "load_dataset_config"]
