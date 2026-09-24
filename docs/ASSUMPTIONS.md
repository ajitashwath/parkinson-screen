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

- **`subject_id_regex` defaults to `null` in `configs/data.yaml`, and stays
  `null` even after real data was integrated.** `extract_subject_id` only
  extracts a subject ID when the caller supplies a working regex; without
  one, `assign_folds` uses the image-hash fallback and `docs/DATASHEET.md`
  reports the leakage caveat loudly, exactly as the phase spec requires.
  HandPD and NewHandPD (added after Phase 1 was first committed; see
  `docs/DECISIONS.md`) get real subject IDs through the separate
  `metadata.csv` mechanism instead, built by
  `training/prepare_handpd.py` from each archive's filename convention, so
  `subject_id_regex` was never needed for them specifically.
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

## Phase 2

- **`tremor_band_ratio`, `peak_freq_hz` and `spectral_entropy` are computed
  from the summed Welch power spectra of the vx and vy velocity
  components, not from a Welch spectrum of the speed (magnitude) signal**,
  even though `CLAUDE.md`'s phrasing ("from a Welch spectrum of the speed
  signal") reads as the latter. Verified empirically (see
  `tremor/features.py`'s `spectral_features` docstring and the recovery
  tests in `tests/test_tremor_features.py`): speed is
  `sqrt(vx^2 + vy^2)`, a nonlinear function of a rotating velocity vector,
  and on a spiral the baseline drawing motion is itself a second, more
  slowly rotating vector. Combining two rotating vectors through a
  magnitude produces intermodulation sidebands at
  `|tremor_freq +/- baseline_rotation_freq|` rather than a clean peak at
  `tremor_freq` (confirmed directly: injecting 10 Hz recovered 9.375 Hz or
  even a doubled ~18.75 Hz harmonic under different tremor-injection
  models, consistently outside the required 0.5 Hz tolerance). Summing the
  linear vx and vy spectra avoids the nonlinearity and recovered every
  tested frequency from 3 to 12 Hz within 0.3 Hz. `mean_speed`, `speed_cv`
  and `jerk_rms` still use the literal speed magnitude, since those are
  time-domain descriptive statistics, not frequency-domain peak-picking,
  and are not affected by this issue.
- **The synthetic tremor injection in `tremor/synthetic.py` adds a small
  circular wobble directly in Cartesian coordinates**
  (`x += A*cos(2*pi*f*t)`, `y += A*sin(2*pi*f*t)`), not a radial
  oscillation in the spiral's own rotating polar frame
  (`radius += A*sin(2*pi*f*t)`). The polar version was tried first and
  produced the exact same intermodulation problem described above, since
  it is still combining two rotating-frame quantities; injecting directly
  in the fixed Cartesian frame avoids that regardless of how the spectral
  features are computed, and matches how tremor is modeled in the
  published spiral-tremor literature.
- **A 2 Hz Butterworth high-pass filter (`remove_voluntary_movement`) is
  applied to vx and vy before the Welch spectrum.** This removes the
  slow, large-amplitude "voluntary movement" component of drawing a
  spiral and is standard tremor-analysis practice; measured to make no
  further difference once the vx/vy-based analysis above was in place
  (the intermodulation was the dominant error source, not low-frequency
  leakage), but kept since it is correct practice and harmless.
- **`spiral_deviation_feature` excludes points within 5% of the trace's
  maximum radius from its centroid** before fitting `r = a + b*theta`.
  The angle `atan2(dy, dx)` is numerically undefined at the origin, so
  including near-center points (present at the start of every spiral)
  inflated the residual even for a perfectly generated ideal spiral
  (measured at approximately 5.5% before this fix, against a chosen
  "should be low" test bound of 5%). This is a real fix, not a loosened
  test: the near-origin angle noise would equally inflate the deviation
  reported for a real user's clean spiral.
- **Tests use a synthetic spiral generator (`tremor/synthetic.py`), not
  real recordings, because no ground-truth tremor frequency exists for
  real recordings without lab equipment this project does not have.**
  This is the one exception to the "no synthetic" data direction from the
  user (see `docs/DECISIONS.md`): CLAUDE.md's own Phase 2 spec requires
  it, and there is no substitute, since the entire point is validating the
  feature extractor against a known frequency that only a generator can
  supply.

## Phase 3

- **Single-logit binary classification (`BCEWithLogitsLoss` on one output),
  not a 2-class softmax head.** The product only ever needs one
  probability (the screening score); a single logit is simpler to carry
  through metrics, ONNX export (Phase 5) and the backend response than a
  2-class head would be, and is mathematically equivalent for a binary
  problem. Recorded since `CLAUDE.md` allows either ("2-class (or
  single-logit) head").
- **`vit_small` uses timm, not Transformers.** `CLAUDE.md` allows either
  "whichever loads reliably"; timm was chosen for consistency with the
  other three CNN/ViT backbones (`resnet50`, `efficientnet_b0`,
  `convnext_tiny`, all timm) and simpler pretrained-weight handling (one
  `create_model` call, no separate image processor object).
  `dinov2_small` uses Transformers specifically, per `CLAUDE.md`.
- **`--fast` loops over every backbone plus the baseline, ignoring the
  config's own `backbone` field**, rather than training only the one
  backbone named in `configs/train_resnet50.yaml`. The Makefile's
  `train-fast` target (fixed in Phase 0) calls `training.train --fast
  --config configs/train_resnet50.yaml` once; the Phase 3 done-criterion
  is that this one command exercises every backbone and the baseline, so
  `--fast` reinterprets "which model" as "all of them, briefly" rather
  than reading it from the config. A real (non-fast) run trains the one
  backbone the config names, once per real fold and seed.
- **`--fast` re-samples a small per-class subset from the real
  HandPD/NewHandPD data and re-assigns fresh folds on that subset**,
  rather than reusing the full dataset's precomputed fold column after
  sampling. The first version sampled globally then filtered by the
  original fold number, which could (and did, empirically) leave the
  single fast-mode test fold with only one class present, making
  ROC-AUC undefined. Verified robust across 10 seeds after the fix; see
  `docs/DECISIONS.md`.
- **`split_train_val`'s internal `n_splits` is capped at the smallest
  class's sample count**, not just at `round(1 / val_fraction)`.
  `StratifiedGroupKFold` cannot stratify past the smallest class's count
  and warns loudly instead; this matters most for `--fast` mode's tiny
  per-fold samples, where the uncapped default (5, for `val_fraction:
  0.2`) could exceed the available minority-class count.
  `bootstrap_auc_ci` and `select_threshold` already degrade gracefully
  (NaN / nearest-available) rather than crashing on small samples, so
  only this one call site needed the fix.
- **Lightning checkpoints are written to a `tempfile.TemporaryDirectory`
  that is deleted immediately after loading the best epoch**, not kept
  under a persistent `checkpoints/` directory. They are working state for
  picking the best epoch, not a deliverable of any single fold or
  `--fast` run (Phase 5's `export_onnx.py` saves the actual champion
  model); keeping every fold's checkpoint around serves no purpose and
  cost real disk space during development (see below).
- **`TEMP`/`TMP` were pointed at a folder on the `E:` drive
  (`E:\parkinson-screen\.tmp`) when running training locally on this
  machine**, because the system default (`C:\Users\...\AppData\Local\Temp`)
  sits on a drive with under 2 GB free, a pre-existing condition on this
  machine unrelated to this project (its own drive, `E:`, has over 10 GB
  free). Checkpoint saving failed with `OSError: No space left on device`
  before this was set, even though the checkpoint's own target directory
  had plenty of room, because an intermediate write landed on `C:` first.
  This is a local environment workaround, not a code change: `make
  train-fast` and CI both still work unmodified on a machine whose default
  temp directory has room.
- **Every `figure.write_html(...)` call passes `include_plotlyjs="cdn"`.**
  The default embeds the full Plotly.js library inline in every HTML
  file (about 3.5 MB each); with 8 report figures committed across
  Phases 2 and 3, that pushed the repository's byte count to roughly 18 MB
  of generated HTML against about 140 KB of actual Python source, which
  made GitHub's language detector report the repo as almost entirely
  HTML. Switching to a CDN reference dropped the same 8 files to about
  45 KB total with no loss of functionality (an internet connection is
  needed to view the charts, same as most Plotly HTML exports already
  assume). `.gitattributes` also marks everything under `reports/` except
  `reports/README.md` as `linguist-generated=true`, so generated output
  is excluded from language statistics regardless of size going forward.
