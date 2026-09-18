"""Unified dataset loader (FR-1).

Loads a dataset configuration from YAML (see ``configs/datasets/``) and a
unified metadata CSV with columns::

    mixture_path, source_1_path, source_2_path, duration

Every audio item is returned as clean 16 kHz mono, either as a
``torch.FloatTensor`` (default) or a ``numpy.ndarray``.

CSV path resolution: a path that is not absolute is resolved against, in
order, the metadata CSV's parent directory, the ``data_root`` from the YAML
config, then the current working directory. The first existing candidate
wins; otherwise the CSV-relative candidate is kept and a clear
``FileNotFoundError`` naming row and field is raised at load time.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import yaml
from torch.utils.data import Dataset

from sepid.data.schema import REQUIRED_UNIFIED_COLUMNS

#: Project-wide audio contract: 16 kHz mono (PRD §9.3 / FR-2).
TARGET_SAMPLE_RATE_HZ: int = 16000

ReturnKind = Literal["torch", "numpy"]


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset configuration parsed from a YAML file.

    Attributes:
        name: Dataset identity (e.g. ``"minilibrimix"``).
        metadata_csv: Path to the unified metadata CSV.
        data_root: Base directory audio paths resolve against.
        target_sample_rate_hz: Output sample rate contract (default 16 kHz).
        mono: Always ``True``; multi-channel audio is downmixed.
        num_sources: Number of reference sources per mixture (default 2).
        split: Frozen partition name (e.g. ``"validation"``).
    """

    name: str
    metadata_csv: Path
    data_root: Path = Path(".")
    target_sample_rate_hz: int = TARGET_SAMPLE_RATE_HZ
    mono: bool = True
    num_sources: int = 2
    split: str = ""


def load_dataset_config(config_path: str | Path) -> DatasetConfig:
    """Load a dataset YAML config into a :class:`DatasetConfig`.

    Args:
        config_path: Path to a YAML file with at least ``metadata_csv``.
            Recognised keys: ``name``, ``metadata_csv``, ``data_root``,
            ``target_sample_rate_hz`` (or legacy ``sample_rate_hz``),
            ``mono``, ``num_sources``, ``split``.

    Returns:
        Parsed :class:`DatasetConfig` with resolved paths.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ValueError: If required keys are missing.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw: Any = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"Dataset config {path} must be a YAML mapping.")
    if "metadata_csv" not in raw:
        raise ValueError(f"Dataset config {path} missing required key 'metadata_csv'.")
    base = path.parent
    metadata_csv = Path(str(raw["metadata_csv"]))
    if not metadata_csv.is_absolute():
        metadata_csv = (base / metadata_csv).resolve()
    data_root = Path(str(raw.get("data_root", ".")))
    if not data_root.is_absolute():
        data_root = (base / data_root).resolve()
    target_sr = int(raw.get("target_sample_rate_hz", raw.get("sample_rate_hz", TARGET_SAMPLE_RATE_HZ)))
    return DatasetConfig(
        name=str(raw.get("name", path.stem)),
        metadata_csv=metadata_csv,
        data_root=data_root,
        target_sample_rate_hz=target_sr,
        mono=bool(raw.get("mono", True)),
        num_sources=int(raw.get("num_sources", 2)),
        split=str(raw.get("split", "")),
    )


def _resolve_audio_path(
    value: str, *, row: int, field: str, csv_parent: Path, data_root: Path
) -> Path:
    """Resolve one CSV audio path against CSV dir, data_root, then CWD."""
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    for base in (csv_parent, data_root, Path.cwd()):
        resolved = base / candidate
        if resolved.is_file():
            return resolved
    # Nothing matched: return the CSV-relative guess so the error names it.
    return csv_parent / candidate


def _load_mono_16k(path: Path, *, target_sr: int, row: int, field: str) -> np.ndarray:
    """Load an audio file as mono float32 at ``target_sr``.

    Args:
        path: Resolved audio file path.
        target_sr: Output sample rate (16 kHz contract).
        row: CSV row index (for error messages).
        field: CSV column name (for error messages).

    Returns:
        1-D ``numpy.ndarray`` of shape ``[T]``, dtype ``float32``.

    Raises:
        FileNotFoundError: If the file does not exist (names row/field).
        ValueError: If the file cannot be decoded.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Row {row}: field '{field}' file not found: {path}")
    try:
        audio, sr = sf.read(str(path), always_2d=False)
    except Exception as exc:
        raise ValueError(f"Row {row}: field '{field}' cannot be decoded ({path}): {exc}") from exc
    mono = np.mean(audio, axis=1) if np.ndim(audio) == 2 else np.asarray(audio)
    mono = np.ascontiguousarray(mono, dtype=np.float32)
    if int(sr) != int(target_sr):
        mono = np.ascontiguousarray(
            librosa.resample(mono, orig_sr=int(sr), target_sr=int(target_sr)),
            dtype=np.float32,
        )
    return mono


class SepIDDataset(Dataset):
    """Unified two-source mixture dataset (FR-1).

    Args:
        config: YAML path, mapping, or :class:`DatasetConfig`.
        return_tensors: ``"torch"`` for ``torch.FloatTensor`` items,
            ``"numpy"`` for ``numpy.ndarray`` items.
        target_sr: Output sample rate override (default 16 kHz).

    Each item is a dict with keys ``mixture``, ``source_1``, ``source_2``
    (each 1-D mono at ``target_sr``), plus ``duration`` (CSV-declared
    seconds) and ``sample_rate``.
    """

    def __init__(
        self,
        config: str | Path | dict[str, Any] | DatasetConfig,
        return_tensors: ReturnKind = "torch",
        target_sr: int = TARGET_SAMPLE_RATE_HZ,
    ) -> None:
        if isinstance(config, (str, Path)):
            self.config: DatasetConfig = load_dataset_config(config)
        elif isinstance(config, dict):
            csv = Path(str(config["metadata_csv"]))
            self.config = DatasetConfig(
                name=str(config.get("name", "inline")),
                metadata_csv=csv,
                data_root=Path(str(config.get("data_root", csv.parent if csv.is_absolute() else "."))),
                target_sample_rate_hz=int(
                    config.get("target_sample_rate_hz", config.get("sample_rate_hz", target_sr))
                ),
                split=str(config.get("split", "")),
            )
        elif isinstance(config, DatasetConfig):
            self.config = config
        else:
            raise TypeError(f"Unsupported config type: {type(config).__name__}")
        if return_tensors not in ("torch", "numpy"):
            raise ValueError("return_tensors must be 'torch' or 'numpy'.")
        self.return_tensors: ReturnKind = return_tensors
        self.target_sr: int = int(self.config.target_sample_rate_hz or target_sr)

        csv_path = self.config.metadata_csv
        if not csv_path.is_file():
            raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")
        frame = pd.read_csv(csv_path)
        missing = [c for c in REQUIRED_UNIFIED_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"Metadata CSV {csv_path} missing columns: {missing}.")
        if frame.empty:
            raise ValueError(f"Metadata CSV {csv_path} contains no rows.")
        csv_parent = csv_path.parent
        self._rows: list[dict[str, Any]] = []
        for idx, record in frame.iterrows():
            row = int(idx) if isinstance(idx, (int, np.integer)) else len(self._rows)
            try:
                duration = float(record["duration"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Row {row}: field 'duration' must be numeric: {record['duration']!r}.") from exc
            if not np.isfinite(duration) or duration <= 0:
                raise ValueError(f"Row {row}: field 'duration' must be positive finite, got {duration}.")
            self._rows.append(
                {
                    "mixture": _resolve_audio_path(
                        str(record["mixture_path"]), row=row, field="mixture_path",
                        csv_parent=csv_parent, data_root=self.config.data_root,
                    ),
                    "source_1": _resolve_audio_path(
                        str(record["source_1_path"]), row=row, field="source_1_path",
                        csv_parent=csv_parent, data_root=self.config.data_root,
                    ),
                    "source_2": _resolve_audio_path(
                        str(record["source_2_path"]), row=row, field="source_2_path",
                        csv_parent=csv_parent, data_root=self.config.data_root,
                    ),
                    "duration": duration,
                }
            )

    def __len__(self) -> int:
        """Return the number of mixtures in the manifest."""
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Load mixture and both sources as 16 kHz mono.

        Args:
            idx: Row index into the metadata CSV.

        Returns:
            Dict with ``mixture``/``source_1``/``source_2`` (1-D mono),
            ``duration`` (CSV seconds), and ``sample_rate``.
        """
        entry = self._rows[int(idx)]
        arrays = {
            key: _load_mono_16k(entry[key], target_sr=self.target_sr, row=int(idx), field=f"{key}_path")
            for key in ("mixture", "source_1", "source_2")
        }
        if self.return_tensors == "torch":
            return {
                key: torch.from_numpy(arr) for key, arr in arrays.items()
            } | {"duration": entry["duration"], "sample_rate": self.target_sr}
        return arrays | {"duration": entry["duration"], "sample_rate": self.target_sr}


__all__ = [
    "DatasetConfig",
    "SepIDDataset",
    "TARGET_SAMPLE_RATE_HZ",
    "load_dataset_config",
]
