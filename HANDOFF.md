# Handoff: what you still have to do

Everything below needs your machine, accounts or people. Nothing here was faked: where a step was
not done, the repository says `TODO(run-on-gpu)` or labels the artifact as a development model.

## 1. Datasets

- [x] HandPD and NewHandPD are downloaded and integrated on the development machine
      (`data/raw/` is gitignored, so a fresh clone does not have them). To rebuild on another
      machine, follow `data/README.md`, then
      `python -m training.prepare_handpd --downloads-dir data/raw/_downloads --out-dir data/raw` and
      `python -m training.audit_data --config configs/data.yaml`.
- [ ] Optional: a labeled kinematic dataset (for example PaHaW) to replace the simulated LightGBM
      training data. It needs a loader and a mapping to `tremor/features.py` inputs.

## 2. GPU runs (Colab or Kaggle)

The final numbers do not exist yet. On a GPU machine, with `requirements-train.txt` installed and
`data/raw/` and `data/folds.csv` in place:

1. `python -m training.train --config configs/train_resnet50.yaml` (5 folds, 3 seeds, all backbones)
2. `python -m training.ablations --config configs/train_resnet50.yaml`
3. `python -m training.shortcut_check --config configs/train_resnet50.yaml`
4. `python -m training.robustness --config configs/train_resnet50.yaml`
5. Pick the champion from the comparison table (do not pick by test fold alone; read
   `docs/DECISIONS.md` on selection), set `backbone` in `configs/export.yaml` and point
   `cv_results_path` at `reports/training_results.json`.
6. `python -m training.explain --config configs/train_resnet50.yaml`
7. `python -m training.register --config configs/export.yaml`
8. `python -m training.export_onnx --config configs/export.yaml` (writes `backend/models/model.onnx`,
   `model_card.json`, `threshold.json`; the parity check must pass)

Copy `reports/` and `backend/models/` back and commit. Then fill the `TODO(run-on-gpu)` rows in
`docs/MODEL_CARD.md` from the generated tables, and rewrite the "what is real" table in `README.md`.
The current `backend/models/model.onnx` is a development placeholder: replace it.

## 3. API keys

- Weights and Biases is optional. With no key, runs use `WANDB_MODE=offline`. To log online, set
  `WANDB_API_KEY` in your environment (never commit it).
- No other keys are needed. Copy `.env.example` to `.env` for local settings.

## 4. Docker and CI (never run yet)

- [ ] Start Docker and run `docker compose up --build`. Open <http://localhost:3000>, draw a spiral,
      submit, check the result and history pages. Fix anything that fails.
- [ ] Record the real backend image size (`docker images`) and peak memory under
      `docker run --memory 512m` in `PROGRESS.md`. The 300 MB peak and the roughly 700 MB image size
      in the notes are a Windows measurement and an estimate.
- [ ] Push to GitHub and check that all four CI jobs pass. The Linux dependency install order in
      `.github/workflows/ci.yml` has not been tested.

## 5. Render deployment

1. Create a Render account and connect the repository (this needs your login).
2. New Blueprint from `render.yaml`.
3. After the first deploy, set `ALLOWED_ORIGINS` on `screening-backend` to the frontend's URL and
   `NEXT_PUBLIC_API_URL` on `screening-frontend` to the backend's URL, then redeploy the frontend.
4. Open the app a minute before any demo: free services sleep when idle.

## 6. Human testing

- [ ] Test-retest with volunteers: the same person draws on the same device on different days.
      Compare tremor features across sessions (see `docs/IDEAS.md`). Get consent, and do not collect
      any diagnosis information unless you have ethics approval.
- [ ] A real browser walkthrough on a touchscreen, a stylus and a mouse, and at phone width. Pointer
      capture and canvas behavior have only been checked through unit tests and a production build.

## 7. Before you present

- Re-read the disclaimer wording on every screen. Nothing should say "diagnose" or "detect
  Parkinson's".
- Do not quote any number from a `--fast` run. `reports/REPORT_ASSETS.md` marks which files are safe.
- Be ready to say that the tremor model is trained on simulated data and that scan-to-canvas domain
  shift is untested (`docs/VIVA_QA.md`).
