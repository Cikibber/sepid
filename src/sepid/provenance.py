"""Run provenance bundle (PRD §11.4: code revision, deps, model, manifest checksums)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class Provenance:
    """Provenance record for one experiment run.

    Attributes:
        run_id: Run identifier (``results/<run_id>/``).
        code_revision: Git SHA or source snapshot hash.
        dependency_lock_hash: Hash of the pinned environment lock.
        model_revision: Frozen checkpoint id + revision.
        manifest_checksum: SHA256 of the frozen manifest.
        started_at: ISO-8601 UTC start timestamp.
    """

    run_id: str
    code_revision: str = "unregistered"
    dependency_lock_hash: str = "unregistered"
    model_revision: str = "unregistered"
    manifest_checksum: str = "unregistered"
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def to_dict(prov: Provenance) -> dict[str, str]:
    """Serialize a :class:`Provenance` record to a plain dict.

    Args:
        prov: Provenance record.

    Returns:
        String-keyed mapping of provenance fields.
    """
    return {k: str(v) for k, v in asdict(prov).items()}


__all__ = ["Provenance", "to_dict"]
