# Assumptions

Defaults picked when the spec in `CLAUDE.md` left something ambiguous, and why.

## Phase 0

- **Heavy training dependencies are not installed on the local dev machine.**
  `requirements-train.txt` pins torch, torchvision, pytorch-lightning, timm and
  transformers as specified, but installing them here would download several
  gigabytes with no GPU to use them on. Per `CLAUDE.md`, real training runs
  happen on Colab or Kaggle GPU notebooks; this repo's `--fast` mode is a CPU
  smoke test that will need those packages once Phase 3 writes real training
  code. Phase 0 verification instead installs and exercises only the dev
  toolchain (ruff, mypy, pytest), which is what `make check` runs. This will
  be revisited if `make setup` needs to succeed in CI (Phase 9), where a Linux
  runner has the time and disk to install the full stack.
- **`.github/workflows/ci.yml`, `docker-compose.yml`, `render.yaml` and the
  Dockerfiles are not created in Phase 0.** The repo layout diagram in
  `CLAUDE.md` section 3 shows the target end state; Phase 0's own task list
  only calls for the Makefile, gitignore, env example, lint/type config,
  progress file, docs stubs and pinned requirements. Those other files are
  explicitly owned by Phase 9 and would be untestable placeholders if written
  now.
- **`backend/`, `training/`, `tremor/` and `common/` get an empty `__init__.py`
  each, but no real modules yet.** This gives ruff and mypy a valid, non-empty
  package to check in Phase 0 without writing ahead of the phase that owns
  each module's real content.
- **Makefile `backend`, `frontend`, `train-fast`, `export` and `up` targets are
  written per the fixed target list in `CLAUDE.md` section 2, but their
  underlying commands will only succeed once the phases that create
  `backend/main.py`, `frontend/package.json`, `training/train.py`,
  `training/export_onnx.py` and `docker-compose.yml` are done. `check` and
  `setup` are the only targets Phase 0 verifies.
- **mypy is run in "light" mode**: `ignore_missing_imports = true` globally,
  since most third-party ML libraries either ship no type stubs or are not
  installed locally yet.
- **A project-local `.venv` is used for all local verification**, created
  with `python -m venv .venv`. `Makefile`'s `setup` target does not create
  the venv itself (it assumes an already-active environment, venv or CI
  container, the standard convention); anyone running `make setup` locally
  should activate a venv first. The first Phase 0 verification pass
  mistakenly installed `ruff`, `mypy` and `pytest` into the machine's global
  Python instead, which downgraded a global `ruff` install below the
  minimum a locally installed `gradio` package required. That was corrected
  immediately (global `ruff` upgraded back to a compatible version) and all
  further work moved into `.venv`, which is already covered by the
  repository's existing `.gitignore`.

## Phase 1

- **`subject_id_regex` defaults to `null` in `configs/data.yaml`.** No
  downloaded dataset's real filename convention is known yet, and
  `CLAUDE.md` rule 2 forbids inventing one. `extract_subject_id` only
  extracts a subject ID when the caller supplies a working regex; until
  then, `assign_folds` always uses the image-hash fallback and
  `docs/DATASHEET.md` will report the leakage caveat loudly, exactly as the
  phase spec requires.
- **A row counts as having a usable subject ID only if every row in the
  manifest has one and there is more than one distinct subject.** A
  partially-populated subject_id column would silently leave some rows
  ungrouped, which is worse than treating the whole dataset as unknown and
  falling back to hash grouping.
- **Fold grouping uses `StratifiedGroupKFold` for both the subject-ID and
  the fallback case**, rather than a separate plain `StratifiedKFold` code
  path for the fallback, as the phase text's wording ("stratified k-fold
  plus duplicate grouping") could be read literally. `StratifiedGroupKFold`
  with `groups=image_hash` already is a stratified k-fold plus duplicate
  grouping; a second, different splitter class would be redundant.
- **Normalization in `common/preprocess.py` only scales to `[0, 1]`**, not
  to ImageNet mean/std. That normalization depends on which pretrained
  backbone Phase 3 picks, so it belongs in the training-time transform, not
  in the shared preprocessing both training and the backend depend on.
- **CPU-only torch and torchvision were installed into the local `.venv`
  for this phase**, to actually run `SpiralWaveDataset` and the
  augmentation pipeline against the synthetic fixture rather than leave
  that test skipped. This is a narrower install than the rest of the
  training stack (no `pytorch-lightning`, `timm`, `transformers`, `mlflow`,
  `wandb`, `lightgbm`, `shap`, none of which Phase 1 code imports), kept
  consistent with the Phase 0 decision to defer the full stack until it is
  actually exercised.
