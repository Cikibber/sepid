# Model card — M2 frozen neural separator (PRD §8.3, FR-5)

Frozen for the main study per the Phase 0 neural-checkpoint decision (OD-2).
Switching after test inspection invalidates direct comparison unless all
conditions are rerun and the change is logged in `docs/DECISIONS.md`.

## Checkpoint identity

| Field | Value |
|---|---|
| Hub ID | `JorisCos/ConvTasNet_Libri2Mix_sepclean_16k` |
| Pinned revision | `e1ef95ab7a037950f3a606b9a56760cf94701d3d` |
| Load string | `JorisCos/ConvTasNet_Libri2Mix_sepclean_16k@e1ef95ab7a037950f3a606b9a56760cf94701d3d` |
| Architecture | ConvTasNet (Asteroid 0.7.0) |
| Training data / task | Libri2Mix `sep_clean` (2-speaker clean separation) |
| Training config | `wav16k/min`, n_src=2, kernel=32, filters=512, stride=16, 8 blocks x 3 repeats |
| Reported test SI-SDR | 15.24 dB (Libri2Mix min test, per Hub README) |

## Runtime contract (enforced by `NeuralSeparator`)

- Sample rate: 16 kHz input required; anything else raises `NeuralSeparatorError`.
- Sources: exactly 2; output shape `[2, time]` at the exact input length (pad/crop).
- Device: CPU inference only in v1 (`device="cpu"`; anything else raises).
- Mode: eval + `torch.no_grad()`; weights frozen, never fine-tuned in the main study.
- Chunking: full-utterance inference in v1 (chunked overlap-add deferred; long-file
  strategy lands with the resource profiler in Epic E).

## License

- Hub tag: `cc-by-sa-4.0`; Hub README states derivative of LibriSpeech (CC BY 4.0),
  released under CC BY-SA 3.0 by the model author.
- Non-commercial training data note (PRD §8.3): LibriSpeech-derived; verify
  institutional/commercial-use constraints before any commercial use.

## Provenance

- Verified 2026-09-18: pinned `@e1ef95a` loads via
  `ConvTasNet.from_pretrained`, eval mode, `(1, 16000)` in → `(1, 2, 16000)` out.
- Note: the checkpoint config omits `sample_rate` in `model_args`, so Asteroid
  reports `model.sample_rate == 8000` (hash-table default); the training README
  and Hub tags confirm 16 kHz, and the adapter enforces 16 kHz input. Do not use
  the object attribute as the sample-rate source of truth.
