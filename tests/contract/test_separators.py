"""Contract test: M0/M1/M2/M3 emit exact shape [2, len(input)] (Phase 2 gate).

Runs all four separators on 2 synthetic audio items (dual-tone mixtures with
disjoint enrollment clips) and asserts deterministic output shapes. M2 uses
the frozen pinned checkpoint (CPU); the test is skipped if `asteroid` is not
installed. M1 runs at reduced K/iterations to keep the gate fast.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sepid.separators.mixture import MixtureBaseline  # noqa: E402
from sepid.separators.nmf_enrolled import EnrolledNMFSeparator  # noqa: E402
from sepid.separators.oracle_mask import OracleMaskSeparator  # noqa: E402

asteroid = pytest.importorskip("asteroid", reason="M2 contract needs asteroid installed")
from sepid.separators.neural import CHECKPOINT_ID, CHECKPOINT_REVISION, NeuralSeparator  # noqa: E402

SR = 16000


def _tone(freq_hz: float, seconds: float, amplitude: float = 0.5) -> np.ndarray:
    """Deterministic sine tone at 16 kHz."""
    t = np.arange(int(SR * seconds), dtype=np.float64) / float(SR)
    return (amplitude * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)


def _items() -> list[dict[str, np.ndarray]]:
    """Two synthetic items: mixtures + disjoint enrollment clips."""
    return [
        {
            "s1": _tone(440.0, 2.0),
            "s2": _tone(660.0, 2.0),
            "e1": _tone(440.0, 1.0, 0.4) + 0.05 * _tone(880.0, 1.0, 0.4),
            "e2": _tone(660.0, 1.0, 0.4) + 0.05 * _tone(990.0, 1.0, 0.4),
        },
        {
            "s1": _tone(523.25, 1.5),
            "s2": _tone(392.0, 1.5),
            "e1": _tone(523.25, 1.0, 0.4) + 0.05 * _tone(1046.5, 1.0, 0.4),
            "e2": _tone(392.0, 1.0, 0.4) + 0.05 * _tone(784.0, 1.0, 0.4),
        },
    ]


def test_frozen_checkpoint_identity() -> None:
    """M2 adapter pins the exact Phase-0 checkpoint id + revision."""
    assert CHECKPOINT_ID == "JorisCos/ConvTasNet_Libri2Mix_sepclean_16k"
    assert CHECKPOINT_REVISION == "e1ef95ab7a037950f3a606b9a56760cf94701d3d"


@pytest.mark.parametrize("item", _items())
def test_m0_output_shape(item: dict[str, np.ndarray]) -> None:
    """M0 duplicates the mixture: [T] -> [2, T], no input mutation."""
    mix = torch.from_numpy((item["s1"] + item["s2"]).astype(np.float32))
    snapshot = mix.clone()
    out = MixtureBaseline().separate(mix)
    assert tuple(out.shape) == (2, mix.numel())
    torch.testing.assert_close(mix, snapshot)


@pytest.mark.parametrize("item", _items())
def test_m1_output_shape(item: dict[str, np.ndarray]) -> None:
    """M1 enrolled NMF: [T] -> [2, T] at exact input length."""
    mix = torch.from_numpy((item["s1"] + item["s2"]).astype(np.float32))
    m1 = EnrolledNMFSeparator(n_components=8, n_iter=20, seed=0)
    out = m1.separate(
        mix,
        torch.from_numpy(item["e1"]),
        torch.from_numpy(item["e2"]),
    )
    assert tuple(out.shape) == (2, mix.numel())
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("item", _items())
def test_m2_output_shape(item: dict[str, np.ndarray]) -> None:
    """M2 frozen ConvTasNet (CPU): [T] -> [2, T], exact length restored."""
    mix = torch.from_numpy((item["s1"] + item["s2"]).astype(np.float32))
    m2 = NeuralSeparator()
    out = m2.separate(mix)
    assert tuple(out.shape) == (2, mix.numel())
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("item", _items())
def test_m3_output_shape(item: dict[str, np.ndarray]) -> None:
    """M3 oracle IRM: [T] + [2, T] refs -> [2, T]."""
    mix = torch.from_numpy((item["s1"] + item["s2"]).astype(np.float32))
    refs = torch.from_numpy(np.stack([item["s1"], item["s2"]]))
    out = OracleMaskSeparator(mask="soft").separate(mix, refs)
    assert tuple(out.shape) == (2, mix.numel())
    assert torch.isfinite(out).all()
