"""Manifest validation: schema errors name the row and field (FR-1)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from sepid.data.schema import REQUIRED_UNIFIED_COLUMNS


def validate_unified_manifest(csv_path: str | Path) -> list[str]:
    """Validate a unified metadata CSV, returning human-readable errors.

    Args:
        csv_path: Path to the metadata CSV.

    Returns:
        List of error strings (empty means valid). Each names row + field.
    """
    path = Path(csv_path)
    if not path.is_file():
        return [f"manifest missing: {path}"]
    frame = pd.read_csv(path)
    errors: list[str] = []
    for col in REQUIRED_UNIFIED_COLUMNS:
        if col not in frame.columns:
            errors.append(f"missing column: '{col}'")
    if errors:
        return errors
    for idx, row in frame.iterrows():
        for col in ("mixture_path", "source_1_path", "source_2_path"):
            if not str(row[col]):
                errors.append(f"Row {idx}: field '{col}' is empty.")
        try:
            duration = float(row["duration"])
        except (TypeError, ValueError):
            errors.append(f"Row {idx}: field 'duration' not numeric: {row['duration']!r}.")
            continue
        if not duration > 0:
            errors.append(f"Row {idx}: field 'duration' must be positive, got {duration}.")
    return errors


__all__ = ["validate_unified_manifest"]
