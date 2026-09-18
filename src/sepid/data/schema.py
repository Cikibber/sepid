"""Manifest schemas: unified loader contract + full PRD §11.1 manifest fields."""
from __future__ import annotations

from dataclasses import dataclass

#: Columns required by the unified metadata CSV consumed by ``SepIDDataset`` (FR-1).
REQUIRED_UNIFIED_COLUMNS: tuple[str, ...] = (
    "mixture_path",
    "source_1_path",
    "source_2_path",
    "duration",
)

#: Full PRD §11.1 manifest fields (one row = one mixture). Nullable reference
#: paths are permitted only for reference-free (D3/naturalistic) data.
FULL_MANIFEST_FIELDS: tuple[str, ...] = (
    "mixture_id",
    "dataset",
    "split",
    "mix_path",
    "source_1_path",
    "source_2_path",
    "sample_rate_hz",
    "num_samples",
    "speaker_1_id",
    "speaker_2_id",
    "transcript_1",
    "transcript_2",
    "language_tags",
    "source_level_ratio_db",
    "overlap_ratio",
    "offset_1_samples",
    "offset_2_samples",
    "session_id",
    "consent_scope",
)


@dataclass(frozen=True)
class MixtureManifestRow:
    """One validated manifest row: mixture + two references + duration.

    Attributes:
        mixture_path: Path to the mixture WAV.
        source_1_path: Path to clean reference 1.
        source_2_path: Path to clean reference 2.
        duration: Mixture duration in seconds (positive, finite).
    """

    mixture_path: str
    source_1_path: str
    source_2_path: str
    duration: float

    def __post_init__(self) -> None:
        """Validate non-empty paths and positive finite duration."""
        for field in ("mixture_path", "source_1_path", "source_2_path"):
            if not str(getattr(self, field)).strip():
                raise ValueError(f"MixtureManifestRow field '{field}' must be non-empty.")
        duration = float(self.duration)
        if not duration > 0:
            raise ValueError(
                f"MixtureManifestRow field 'duration' must be positive, got {self.duration!r}."
            )

    def as_dict(self) -> dict[str, object]:
        """Serialize the row to a CSV-ready mapping.

        Returns:
            Mapping with keys matching ``REQUIRED_UNIFIED_COLUMNS``.
        """
        return {
            "mixture_path": self.mixture_path,
            "source_1_path": self.source_1_path,
            "source_2_path": self.source_2_path,
            "duration": float(self.duration),
        }


__all__ = ["FULL_MANIFEST_FIELDS", "MixtureManifestRow", "REQUIRED_UNIFIED_COLUMNS"]
