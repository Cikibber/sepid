# SepID — Single-Channel Speech Separation for Indonesian Group Discussions

Course final project (Digital Signal Processing). Full contract: `PRD_SepID_v1.0_completed.md`.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Smoke check

```powershell
python -m unittest discover -s tests -v
```

## Layout (§11)

- `configs/` — dataset, method, experiment YAMLs
- `manifests/` — metadata only (frozen versions, no audio)
- `src/sepid/` — canonical package; `src/data_loader.py` and
  `src/preprocessing.py` are compatibility shims re-exporting it
- `tests/` — unit, contract, integration
- `results/figures/` — generated plots (gitignored audio elsewhere)
