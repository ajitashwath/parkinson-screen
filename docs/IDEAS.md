# Ideas (out of scope)

Things that came up while building that are not in `CLAUDE.md` and are not
being built. Recorded here instead of built, per rule 5 in `CLAUDE.md`.

- Calibrate the image model's probabilities (temperature scaling or isotonic
  regression) on a held-out validation split, then report reliability curves.
- Add PaHaW or another labeled kinematic dataset and retrain the LightGBM
  tremor model on real recordings instead of simulated ones.
- Test-retest reliability study: have volunteers draw on the same device on
  several days and report the intraclass correlation of each tremor feature.
- Evaluate the image model on canvas drawings from volunteers (no diagnosis
  labels needed to measure how far scores shift from the scan domain).
- A benchmark script for the backend (latency and peak memory) committed
  under `backend/`, so deployment numbers are reproducible.
- Server-side image-model ensembling across folds instead of one refit model.
- Persistent storage and accounts for longitudinal history across devices.
