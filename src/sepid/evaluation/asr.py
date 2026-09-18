"""ASR evaluation skeleton (Epic E; frozen normalization + ORC-WER/cpWER pilot)."""
from __future__ import annotations


def normalize_transcript(text: str) -> str:
    """Frozen transcript normalization (decided at Phase 3 gate; stub)."""
    raise NotImplementedError("ASR evaluation lands in Epic E.")


__all__ = ["normalize_transcript"]
