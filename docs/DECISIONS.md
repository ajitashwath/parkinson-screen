# Decisions

Engineering decisions made along the way, with the reasoning, so later phases
do not silently contradict earlier ones.

## Phase 0

- **Two dependency files, never merged.** `requirements-train.txt` for
  training and dev tooling, `backend/requirements.txt` for the FastAPI
  service. No torch and no `shap` package in the backend file, per
  `CLAUDE.md` section 2, so the backend Docker image stays small enough for
  Render's free tier.
- **Tool config lives in one `pyproject.toml`** (`[tool.ruff]`, `[tool.mypy]`,
  `[tool.pytest.ini_options]`) rather than separate `ruff.toml` / `mypy.ini`
  files, to keep Phase 0 config in one place.
- **Lint and type-check scope excludes `data/`, `frontend/`, `reports/`,
  `mlruns/` and `wandb/`.** Those hold generated artifacts, JS/TS code (its
  own ESLint/tsc will cover it from Phase 8), or third-party output, not
  first-party Python.

## Phase 1 (revised after Phase 1 was first committed)

- **Claude Code downloaded real datasets directly, superseding the
  original `HANDOFF.md`-style plan of leaving all downloads to the user.**
  `CLAUDE.md`'s own "Things you must do yourself" section originally listed
  dataset downloads under things Claude Code cannot do, largely because
  Kaggle requires an account and login. After Phase 1 was committed with
  only synthetic test fixtures, the user explicitly asked for real data to
  be researched and integrated. Claude Code researched available sources
  and found HandPD and NewHandPD (UNESP, Botucatu Medical School) served
  directly over plain HTTP with no login, account, or request form, so
  downloading them does not conflict with the standing rule against
  creating accounts or entering credentials. The user was shown the exact
  files, source, and total size (about 340 MB) and confirmed before any
  download happened. This does not extend to Kaggle's "Parkinson's
  Drawings" dataset or any other source that requires an account: those
  remain the user's own task, per `data/README.md`.
- **Subject IDs now come from a `metadata.csv` mechanism in
  `training/data.py`, checked before falling back to a caller-supplied
  filename regex.** The original plan was to build that `metadata.csv`
  straight from HandPD's and NewHandPD's own published CSVs
  (`ID_PATIENT` column). After downloading the real archives, both turned
  out to be unusable for this: HandPD's `Spiral_HandPD.csv` /
  `Meander_HandPD.csv` are plain LibOPF feature exports with no image name
  or patient ID column at all, despite the dataset page describing richer
  ones, and NewHandPD's `NewSpiral.csv` has a publishing bug where every
  patient row's `IMAGE_NAME` points at a meander file regardless of which
  CSV it is in (confirmed by joining the CSV against the real archive:
  zero of 124 real patient spiral filenames appear in it), and its
  `ID_PATIENT` overlaps between the healthy and patient classes. Both
  archives instead encode a real, dataset-native subject code directly in
  every filename with no exceptions (`0068-1.jpg` for HandPD, `sp1-H9.jpg`
  / `mea1-H9.jpg` for NewHandPD, verified to agree across the spiral and
  meander archives for the same person). `training/prepare_handpd.py`
  derives `subject_id` from the filename instead of the CSVs and writes
  `data/raw/<dataset>/metadata.csv` from that. The `metadata.csv`
  mechanism itself in `training/data.py` stays general (any future dataset
  can supply one), only HandPD's own derivation changed.

## Phase 2

- **The user's "no synthetic" direction does not apply to
  `tremor/synthetic.py`.** That instruction was about not settling for
  synthetic stand-ins where real data was available and gettable (Phase
  1's images); `CLAUDE.md`'s own Phase 2 spec requires a synthetic spiral
  generator as the actual methodology, since no real recording has a
  ground-truth tremor frequency to validate against without lab equipment
  this project does not have. See `docs/ASSUMPTIONS.md` for the full
  reasoning and the signal-processing decisions this phase needed to
  actually hit the required 0.5 Hz recovery accuracy.

## Phase 3

- **`--fast` trains real HandPD/NewHandPD images (a small sampled
  subset), not synthetic ones**, consistent with the same real-data
  direction from Phase 1. The only exception anywhere in this project
  remains Phase 2's tremor synthetic generator, which has no real-data
  substitute. Full reasoning for the `--fast` design (looping over every
  backbone, re-sampling then re-folding) in `docs/ASSUMPTIONS.md`.

## Phase 5

- **The committed `backend/models/model.onnx` is a development model.**
  Phase 7's backend needs a model file to load and Phase 5's done
  criterion allows a tiny fast-mode model "clearly labeled as such". The
  card carries `development_model: true` and a warning, and
  `threshold.json` sits beside it. `model_int8.onnx` is gitignored because
  it regenerates from the same command and binary churn in history is
  worth avoiding. Replacing the dev model with a real champion is a
  `HANDOFF.md` item.

## Phase 6

- **Contributions come from LightGBM's `pred_contrib=True`, not the
  `shap` package.** It is exact TreeSHAP, keeps the backend image free of
  shap and matplotlib, and a test asserts rows sum to the raw score.
- **`shap` and matplotlib are imported lazily inside
  `save_shap_figures`,** so importing `tremor.train_lgbm` stays light.
- **Synthetic-only training is stated in the card, the provenance string
  and the docs,** because the spec requires the backend and UI to show it.
