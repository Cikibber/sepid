"""Acquire MiniLibriMix validation split (Epic D, PRD §9.2).

Zenodo DOI 10.5281/zenodo.3871592 (MiniLibriMix.zip, ~640 MB,
md5:32bda08f4992f584f5703c46ff2321db). Downloads + verifies only when the
local copy is absent or corrupt, extracts, then writes
``manifests/minilibrimix_val.csv`` with one :class:`MixtureManifestRow`
per ``mix_clean`` dev mixture (mixture + s1/s2 references + duration).

Usage:
    python scripts/acquire_minilibrimix.py [--data-root data/minilibrimix]
        [--manifest manifests/minilibrimix_val.csv] [--skip-download]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402

from sepid.data.schema import MixtureManifestRow  # noqa: E402
from sepid.data.validation import validate_unified_manifest  # noqa: E402

#: Zenodo artifact (PRD §9.2: official v1.0, DOI 10.5281/zenodo.3871592).
ZENODO_URL = "https://zenodo.org/records/3871592/files/MiniLibriMix.zip?download=1"
ARCHIVE_NAME = "MiniLibriMix.zip"
EXPECTED_MD5 = "32bda08f4992f584f5703c46ff2321db"
EXPECTED_SIZE = 640547371


def md5_of(path: Path, chunk: int = 1 << 20) -> str:
    """Stream MD5 hex digest of a file.

    Args:
        path: File to hash. chunk: Read size in bytes.

    Returns:
        Lowercase hex MD5 digest.
    """
    digest = hashlib.md5()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def download_archive(dest: Path) -> Path:
    """Download the Zenodo archive with checksum verification.

    Args:
        dest: Directory holding ``MiniLibriMix.zip``.

    Returns:
        Path to the verified archive.

    Raises:
        ValueError: If the MD5 checksum mismatches.
    """
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / ARCHIVE_NAME
    if archive.is_file() and md5_of(archive) == EXPECTED_MD5:
        print(f"Archive present and verified: {archive}")
        return archive
    print(f"Downloading {ZENODO_URL} (~{EXPECTED_SIZE / 1e6:.0f} MB) ...")
    req = urllib.request.Request(ZENODO_URL, headers={"User-Agent": "Mozilla/5.0 (SepID)"})
    with urllib.request.urlopen(req, timeout=120) as resp, archive.open("wb") as fh:
        while True:
            block = resp.read(1 << 20)
            if not block:
                break
            fh.write(block)
    digest = md5_of(archive)
    if digest != EXPECTED_MD5:
        raise ValueError(f"Checksum mismatch for {archive}: got {digest}.")
    print(f"Download verified (md5:{digest}): {archive}")
    return archive


def extract_archive(archive: Path, data_root: Path) -> Path:
    """Extract the archive under ``data_root`` (idempotent marker file).

    Args:
        archive: Verified zip path. data_root: Extraction directory.

    Returns:
        The dataset root directory.
    """
    marker = data_root / ".extracted"
    if marker.is_file():
        print(f"Already extracted: {data_root}")
        return data_root
    print(f"Extracting {archive} -> {data_root} ...")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(data_root)
    marker.write_text(f"{ARCHIVE_NAME} md5:{EXPECTED_MD5}\n", encoding="utf-8")
    return data_root


def find_split_dir(data_root: Path) -> Path:
    """Locate the validation ``mix_clean`` directory under ``data_root``.

    Searches common MiniLibriMix layouts (``wav16k/min/dev/mix_clean`` or
    any nested ``dev/mix_clean``) and returns the first match.

    Args:
        data_root: Extracted dataset root.

    Returns:
        Path to the ``mix_clean`` directory.

    Raises:
        FileNotFoundError: If no validation ``mix_clean`` directory exists.
    """
    candidates = list(data_root.rglob("dev/mix_clean"))
    if not candidates:
        raise FileNotFoundError(f"No dev/mix_clean directory under {data_root}.")
    split = sorted(candidates)[0]
    print(f"Validation split dir: {split}")
    return split


def build_manifest(split_dir: Path, manifest_path: Path) -> Path:
    """Write one :class:`MixtureManifestRow` per mix_clean mixture.

    Matches each ``mix_clean/<id>.wav`` with ``s1/<id>.wav`` and
    ``s2/<id>.wav`` in sibling directories and records soundfile duration.

    Args:
        split_dir: The ``mix_clean`` directory.
        manifest_path: Destination CSV path.

    Returns:
        The manifest path.

    Raises:
        FileNotFoundError: If a mixture lacks matching s1/s2 references.
        ValueError: If manifest validation reports errors.
    """
    s1_dir, s2_dir = split_dir.parent / "s1", split_dir.parent / "s2"
    mixes = sorted(split_dir.glob("*.wav"))
    if not mixes:
        raise FileNotFoundError(f"No mixtures in {split_dir}.")
    rows: list[MixtureManifestRow] = []
    for mix in mixes:
        s1, s2 = s1_dir / mix.name, s2_dir / mix.name
        missing = [p for p in (s1, s2) if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Mixture {mix.name} missing references: {missing}.")
        info = sf.info(str(mix))
        rows.append(
            MixtureManifestRow(
                mixture_path=str(mix),
                source_1_path=str(s1),
                source_2_path=str(s2),
                duration=float(info.frames) / float(info.samplerate),
            )
        )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r.as_dict() for r in rows]).to_csv(manifest_path, index=False)
    errors = validate_unified_manifest(manifest_path)
    if errors:
        raise ValueError(f"Manifest validation failed: {errors}")
    print(f"Wrote {len(rows)} rows -> {manifest_path} (validation clean).")
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: download, extract, and build the validation manifest.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Acquire MiniLibriMix validation split.")
    parser.add_argument("--data-root", default="data/minilibrimix")
    parser.add_argument("--manifest", default="manifests/minilibrimix_val.csv")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args(argv)
    data_root = (REPO_ROOT / args.data_root) if not Path(args.data_root).is_absolute() else Path(args.data_root)
    manifest = (REPO_ROOT / args.manifest) if not Path(args.manifest).is_absolute() else Path(args.manifest)
    if not args.skip_download:
        archive = download_archive(data_root)
        extract_archive(archive, data_root)
    build_manifest(find_split_dir(data_root), manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
