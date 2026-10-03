# Progress

Phase checklist. Updated after every phase with what was done, what was
verified (with the exact command run), and any open issue. See `CLAUDE.md`
for the full spec.

## Phase 0: Scaffold - done

**Plan:** create the repo layout, Makefile targets, gitignore, env example,
ruff/mypy config, PROGRESS.md, docs stubs, and pinned dependency files. Done
when `make setup && make check` passes with one placeholder test.

**Built:**
- `CLAUDE.md` (this project's full spec, saved from the builder prompt)
- Directories: `docs/`, `configs/`, `data/`, `common/`, `training/`,
  `tremor/`, `backend/`, `tests/`, `reports/`, `frontend/`
- `common/__init__.py`, `training/__init__.py`, `tremor/__init__.py`,
  `backend/__init__.py` (empty packages, real content starts in their
  respective phases)
- `tests/test_placeholder.py`
- `configs/README.md`, `data/README.md`, `reports/README.md`,
  `frontend/README.md` (stubs for directories not yet populated)
- `Makefile` with targets: `setup`, `check`, `lint`, `typecheck`, `test`,
  `train-fast`, `export`, `backend`, `frontend`, `up`
- `pyproject.toml` with `[tool.ruff]`, `[tool.mypy]`,
  `[tool.pytest.ini_options]`
- `.env.example`
- `.gitignore` extended with project-specific entries (raw/processed data,
  mlruns, wandb, node_modules, .next, local db files)
- `requirements-train.txt` (training stack + dev tools, pinned)
- `backend/requirements.txt` (FastAPI serving stack, pinned, no torch, no
  `shap` package)
- `docs/ASSUMPTIONS.md`, `docs/DECISIONS.md`, `docs/IDEAS.md`,
  `docs/MODEL_CARD.md`, `docs/DATASHEET.md`, `docs/VIVA_QA.md`

**Verified:**
- `python -m venv .venv` - created a project-local virtual environment (see
  `docs/ASSUMPTIONS.md`: the first attempt installed the dev toolchain into
  the machine's global Python by mistake and was corrected).
- `.venv/Scripts/python.exe -m pip install ruff==0.6.2 mypy==1.11.1
  pytest==8.3.2 PyYAML==6.0.2` - installed cleanly into `.venv` (the
  dev-tool subset of `requirements-train.txt`; see `docs/ASSUMPTIONS.md` for
  why the full heavy ML stack was not installed locally in this phase).
- `.venv/Scripts/python.exe -m ruff check .` - "All checks passed!"
- `.venv/Scripts/python.exe -m mypy common training tremor backend tests`
  ran and reported: "Success: no issues found in 5 source files"
- `.venv/Scripts/python.exe -m pytest -q` - "1 passed"
  (`tests/test_placeholder.py`).
- `make` itself is not installed on this Windows dev machine (confirmed via
  `which make`), so the three commands above were run directly rather than
  through `make check`. The Makefile targets call exactly these commands, so
  this verifies the same logic `make check` will run once `make` (or CI's
  Linux runner) executes it. `.github/workflows/ci.yml` is added in Phase 9
  and will run `make check` on Linux where `make` is present.

**Open:**
- `make setup` is not yet exercised end to end (full training stack
  install). Deferred until a phase actually needs those packages installed,
  per `docs/ASSUMPTIONS.md`.
- `make` itself is unavailable locally; targets are verified via their
  underlying shell commands instead.

**Assumptions:** see `docs/ASSUMPTIONS.md`.

**Next:** Phase 1 (data layer and preprocessing). Needs from you: nothing to
start the code; before Phase 1's audit can run on real data you will need to
download the datasets into `data/raw/` per the instructions Phase 1 writes to
`data/README.md`.

---

## Phase 1: Data layer and preprocessing - done

**Plan:**
- `common/preprocess.py`: the single deterministic preprocessing pipeline
  (grayscale, crop to content bounding box with padding, resize with white
  padding to a square, normalize to [0, 1], replicate to 3 channels, plus a
  `skeleton_rerender` mode). Used by training now and by the backend from
  Phase 7 onward.
- `training/data.py`: `discover_images` builds a manifest from
  `data/raw/<dataset>/<spiral|wave>/<healthy|parkinson>/*.png`;
  `extract_subject_id` only recognizes a caller-supplied regex (no invented
  filename convention, per `CLAUDE.md` rule 2); `assign_folds` does a
  grouped stratified k-fold, grouping by subject ID when every row has one,
  else falling back to grouping by perceptual image hash so duplicates
  never split across folds; `build_train_augmentation` and
  `SpiralWaveDataset` for train-only, on-the-fly augmentation.
- `training/audit_data.py`: CLI that runs `discover_images` +
  `assign_folds`, prints the summary, and can write it to a datasheet file.
- `configs/data.yaml`: raw_dir, image size, padding, fold count, seed,
  subject_id_regex (null by default).
- `data/README.md`: full download instructions and the expected directory
  layout for Kaggle "Parkinson's Drawings", HandPD/NewHandPD and PaHaW.
- Tests: `tests/conftest.py` (synthetic spiral image factory and a fake
  `data/raw/` tree fixture, so nothing depends on a real licensed dataset),
  `tests/test_preprocess.py`, `tests/test_data.py`.

**Done when:** audit runs end to end on the synthetic fixture, fold
assignment has zero subject or duplicate-hash overlap across folds
(tested), and preprocessing tests pass.

**Built:**
- `common/preprocess.py`: `load_grayscale`, `crop_to_content`,
  `resize_with_padding`, `normalize`, `to_three_channel`,
  `skeleton_rerender`, `preprocess_image` (the orchestrating entry point),
  `PreprocessConfig`.
- `training/data.py`: `discover_images`, `extract_subject_id`,
  `compute_image_hash`, `has_usable_subject_ids`, `summarize`,
  `assign_folds`, `save_folds_csv`, `build_train_augmentation`,
  `SpiralWaveDataset`.
- `training/audit_data.py`: CLI (`--config`, `--seed`, `--fast`, `--raw-dir`,
  `--out`, `--folds-out`), plus `render_datasheet_markdown` and
  `sample_for_fast_mode`.
- `configs/data.yaml`.
- `data/README.md` rewritten with real dataset guidance (Kaggle
  "Parkinson's Drawings", HandPD/NewHandPD, PaHaW), the expected
  `data/raw/<dataset>/<spiral|wave>/<healthy|parkinson>/*.png` layout, and
  the leakage-fallback caveat.
- `docs/DATASHEET.md` turned into a pointer at the audit command (real
  content only comes from a real dataset, so this stays a TODO until one is
  downloaded).
- Tests: `tests/conftest.py` (`spiral_image_factory`, `blank_image`,
  `synthetic_raw_dataset`, plus a `draw_wave` helper), `tests/test_preprocess.py`
  (17 tests), `tests/test_data.py` (19 tests, including a
  `pytest.importorskip("torch")`-guarded test of `SpiralWaveDataset`).

**Verified:**
- Installed `numpy`, `pandas`, `scikit-learn`, `scikit-image`,
  `opencv-python-headless`, `Pillow`, `imagehash`, `PyYAML`, `scipy` into
  `.venv`, plus CPU-only `torch`/`torchvision` from
  `https://download.pytorch.org/whl/cpu` (so `SpiralWaveDataset` and the
  augmentation pipeline are actually exercised, not skipped) and
  `types-PyYAML` for mypy.
- `.venv/Scripts/python.exe -m ruff check .` - "All checks passed!"
- `.venv/Scripts/python.exe -m mypy common training tremor backend tests` -
  "Success: no issues found in 11 source files"
- `.venv/Scripts/python.exe -m pytest -q` - 36 passed (35 real, 1 no longer
  skipped once torch was installed).
- `.venv/Scripts/python.exe -m training.audit_data --config configs/data.yaml
  --raw-dir <scratch> --out <scratch> --folds-out <scratch> --seed 42` run
  against a scratch synthetic dataset (20 images, 4 kind/label groups): printed
  the expected counts, fold method, image sizes and duplicate/subject-ID
  facts, and wrote a readable datasheet and folds CSV. Real `data/` and
  `docs/DATASHEET.md` were left untouched by this run.

**Bugs found and fixed during verification (not just written and assumed
correct):**
- The first version of `synthetic_raw_dataset` drew every "wave" image with
  the same spiral shape as "spiral" images (only extra jitter noise), so
  `imagehash.phash` collapsed structurally-similar-but-unrelated images
  into false duplicate groups. Fixed by adding a real `draw_wave` (sine
  curve) helper and varying turns/cycles by kind, label and index, so only
  the deliberately duplicated file collides. Caught by
  `test_discover_images_flags_duplicate_hash` and
  `test_summarize_reports_expected_fields` actually failing on first run.
- `test_discover_images_flags_duplicate_hash` itself had an ambiguous
  filename substring check (`"healthy_00"` matched both the spiral and the
  wave folder); narrowed it to filter by `kind` and `label` first.
- `training/audit_data.py` used `python training/audit_data.py`, which
  fails with `ModuleNotFoundError: No module named 'training'` because the
  script's own directory, not the repo root, ends up on `sys.path`. Fixed
  to `python -m training.audit_data` everywhere (script docstring,
  `data/README.md`, `docs/DATASHEET.md`) and corrected the same pattern
  preemptively in the Makefile's `train-fast` and `export` targets for
  Phase 3 and Phase 5.
- The first real CLI run used the config's default `folds_path`
  (`data/folds.csv`) while pointing `--raw-dir` at a scratch synthetic
  fixture, which wrote fake fold data into the real `data/` directory.
  Caught immediately by checking `data/` after the run; deleted the file
  and added a `--folds-out` CLI override (symmetric with the existing
  `--out`) so a demo or test run can never again write into the real
  `data/` path by default.

**Open:** none for this phase.

**Assumptions:** see `docs/ASSUMPTIONS.md` (Phase 1 section).

---

## Phase 1 addendum: real dataset integration

After the above was committed, the user explicitly asked for real data to
be researched and integrated rather than left as a `data/README.md`
instruction for later. Full reasoning in `docs/DECISIONS.md`.

**Built:**
- `training/prepare_handpd.py`: downloads-to-canonical-layout ETL for
  HandPD and NewHandPD. `is_junk_member` filters macOS `.DS_Store` and
  AppleDouble `._*` sidecar files that share the real `.jpg` extension and
  would otherwise be picked up as images. `extract_subject_id` pulls the
  dataset-native subject code straight from each filename.
  `prepare_handpd`/`prepare_newhandpd` extract the zip archives, sort
  images into `<spiral|wave>/<healthy|parkinson>/`, and write
  `metadata.csv`.
- `tests/test_prepare_handpd.py`: 12 tests against tiny synthetic
  in-memory zip fixtures (junk filtering, subject ID regexes, archive
  extraction, control/patient path-based labeling).
- Rewrote `data/README.md` to describe what was actually downloaded, exact
  reproduction commands, and the CSV bugs found (below).
- Regenerated `docs/DATASHEET.md` and `data/folds.csv` for real, from real
  data.
- Improved `render_datasheet_markdown` and the CLI's own print in
  `training/audit_data.py` to summarize image sizes (761 distinct sizes in
  the real data) instead of dumping every one, and to explain, when folds
  are subject-grouped, why a high duplicate-hash count is expected and not
  a leakage risk (repeated attempts of the same printed template hash as
  near-identical).

**Verified:**
- Downloaded HandPD (`Spiral_HandPD.zip`, `Meander_HandPD.zip`,
  132 MB + 90 MB) and NewHandPD (4 zips, 116 MB) from
  `wwwp.fc.unesp.br` after explicit user confirmation of the exact files,
  source and size. Every file's size matched the server's `Content-Length`
  exactly after a resume (`curl -C -`) following a `--max-time` truncation
  on the first attempt.
  `.venv/Scripts/python.exe -m training.prepare_handpd --downloads-dir
  data/raw/_downloads --out-dir data/raw` - wrote 736 HandPD images and
  528 NewHandPD images, matching the dataset page's stated counts (92
  subjects x 4 spiral + 4 meander = 736; 66 subjects x 4 + 4, minus the
  macOS junk files, = 528).
  `.venv/Scripts/python.exe -m training.audit_data --config
  configs/data.yaml --seed 42` on the real combined data - 1264 images,
  `fold_method: grouped_by_subject_id`, `subject_ids_available: True`.
  Verified in a one-off script that of 158 unique subjects across both
  datasets, 0 span more than one fold.
- `.venv/Scripts/python.exe -m ruff check .` - "All checks passed!"
- `.venv/Scripts/python.exe -m mypy common training tremor backend tests`
  - "Success: no issues found in 13 source files"
- `.venv/Scripts/python.exe -m pytest -q` - 52 passed.

**Bugs found and fixed during verification:**
- The first download attempt used `curl --max-time 300`, too short for
  the 132 MB `Spiral_HandPD.zip` on this slow server; it failed at 67 MB
  (`curl` exit 28, timeout). Resumed with `curl -C - --max-time 1800`.
- HandPD's own `Spiral_HandPD.csv` / `Meander_HandPD.csv`, which the
  dataset page describes as having `IMAGE_NAME` and `ID_PATIENT` columns,
  turned out on inspection to be plain LibOPF feature exports
  (`368,2,9` header, i.e. sample/class/feature counts, no names or IDs at
  all). Caught by actually loading the file with pandas instead of trusting
  the page's description.
- NewHandPD's `NewSpiral.csv` has a publishing bug: joining its
  `IMAGE_NAME` column against the real extracted archive contents showed
  zero of the 124 real patient spiral filenames appear in it (every patient
  row's `IMAGE_NAME` instead points at a meander file), and its
  `ID_PATIENT` has 2 values shared between the healthy and patient classes,
  which a real per-class code should not have. Caught by the same
  join-against-real-files check, not assumed from the CSV alone.
  `training/prepare_handpd.py` was written to derive `subject_id` from
  each archive's own filename convention instead (verified consistent
  across the spiral and meander archives for every subject, with zero
  healthy/patient collisions), bypassing both CSV problems entirely.
- The first version of `render_datasheet_markdown` dumped every distinct
  image size as one long comma-separated line; real scanned data produced
  761 distinct sizes, an unreadable ~3000-character line. Changed to a
  width/height range summary once there are more than 12 distinct sizes.

**Open:** none.

**Assumptions:** see `docs/ASSUMPTIONS.md` and `docs/DECISIONS.md` (Phase 1
sections).

**Next:** Phase 2 (tremor analysis package). Needs from you: nothing to
start the code.

---

## Phase 2: Tremor analysis package - in progress

**Plan:**
- `tremor/features.py`: `StrokePoint` dataclass (x, y, t_ms, pressure,
  pointer_type, pen_lift); `validate_points` returns warnings (never
  raises) for short duration, too few points, or low sampling rate;
  `resample_uniform` interpolates to 100 Hz; `compute_features` smooths
  with Savitzky-Golay then derives speed/acceleration/jerk and the full
  feature set from CLAUDE.md Phase 2 (duration_s, path_length, mean_speed,
  speed_cv, jerk_rms, pen_lifts, pause_ratio, tremor_band_ratio,
  peak_freq_hz, spectral_entropy, spiral_deviation, pressure_mean/sd).
- `tremor/synthetic.py`: `generate_synthetic_spiral` builds an ideal
  Archimedean spiral trajectory, optionally injecting a band-limited radial
  oscillation at a chosen frequency and amplitude (the tremor proxy), plus
  configurable sampling jitter and a mouse-noise mode.
- `tremor/validate.py`: uses the synthetic generator as ground truth to
  plot injected vs recovered `peak_freq_hz`, tremor amplitude vs
  `tremor_band_ratio`, and feature stability under jitter/mouse noise;
  saves Plotly HTML to `reports/`.
- Tests: recovered frequency within 0.5 Hz of injected across 3 to 12 Hz;
  `tremor_band_ratio` increases with amplitude; short/sparse input returns
  warnings not exceptions; features are deterministic for the same input.

**Done when:** all tremor tests pass and the validation figures are
generated by actually running `tremor/validate.py`.

**Built:**
- `tremor/features.py`: `StrokePoint`, `validate_points`,
  `resample_uniform`, `smooth`, `compute_kinematics`, `spectral_features`,
  `remove_voluntary_movement`, `spiral_deviation_feature`,
  `count_pen_lifts`, `compute_features` (the full feature set from
  `CLAUDE.md` Phase 2).
- `tremor/synthetic.py`: `generate_synthetic_spiral`.
- `tremor/validate.py`: `frequency_recovery_data`, `amplitude_response_data`,
  `stability_data`, the three Plotly figure builders, `run_validation`,
  and a CLI (`--config`, `--seed`, `--fast`).
- `configs/tremor.yaml`.
- Tests: `tests/test_tremor_features.py` (20 tests, parametrized frequency
  recovery across 3, 5, 6.5, 8, 10, 12 Hz), `tests/test_tremor_validate.py`
  (5 tests, including one that writes real files and checks they exist and
  are non-trivial in size).

**Verified:**
- Installed `plotly` into `.venv` (already pinned in
  `requirements-train.txt`, not previously installed locally).
- `.venv/Scripts/python.exe -m ruff check .` - "All checks passed!"
- `.venv/Scripts/python.exe -m mypy common training tremor backend tests`
  - "Success: no issues found in 18 source files"
- `.venv/Scripts/python.exe -m pytest -q` - 77 passed.
- `.venv/Scripts/python.exe -m tremor.validate --config configs/tremor.yaml`
  run for real: "Frequency recovery: max error 0.281 Hz across [3.0, 4.0,
  5.0, 6.0, 6.5, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0] Hz", "Amplitude response:
  tremor_band_ratio 0.000 to 0.999", and wrote
  `reports/tremor_frequency_recovery.html`,
  `reports/tremor_amplitude_response.html`, `reports/tremor_stability.html`
  (each about 3.6 MB, real Plotly figures, confirmed non-empty).

**Bugs found and fixed during verification, not assumed correct from the
spec text alone:**
- The first tremor-injection model (radial modulation in the spiral's own
  rotating polar frame) produced frequency-doubling: injecting 5 Hz
  recovered 10.15625 Hz. Root cause: combining two rotating-frame
  quantities (the spiral's own rotation and the injected radial
  oscillation) through the nonlinear position/velocity relationship
  produces harmonics. Fixed by injecting the tremor as a direct Cartesian
  circular wobble instead.
- Even with that fix, computing the Welch spectrum of the speed
  (magnitude) signal still produced errors up to 0.75 Hz, because a
  perfectly circular wobble has *constant* speed magnitude in isolation
  (all its signal is in direction, not magnitude); combined with a second,
  slower-rotating baseline vector (the spiral's own motion), the magnitude
  of the sum oscillates at the beat frequency
  `|tremor_freq - baseline_rotation_freq|`, not at `tremor_freq` itself.
  Root-caused by direct experimentation (a small script printing the top
  power frequencies), not guessed. Fixed by summing the Welch spectra of
  the vx and vy velocity components instead, which are linear and do not
  produce this intermodulation; every tested frequency 3 to 12 Hz then
  recovered within 0.3 Hz.
- `spiral_deviation_feature` reported about 5.5% deviation for a
  perfectly generated, noise-free ideal spiral, against a chosen "should
  be low" test bound of 5%. Root cause: `atan2` is numerically undefined
  at the origin, and every spiral starts there, so the handful of
  near-center points were distorting the linear fit. Fixed by excluding
  points within 5% of the trace's maximum radius from the fit, which
  reduced the deviation and is also more correct for real drawings (the
  same near-origin noise would otherwise inflate the reported deviation
  for a real user's clean spiral too).

**Open:** none.

**Assumptions:** see `docs/ASSUMPTIONS.md` and `docs/DECISIONS.md` (Phase
2 sections) for the full reasoning behind the spectral analysis approach.

**Next:** Phase 3 (training pipeline). Needs from you: nothing to start
the code; a GPU (Colab or Kaggle) will be needed for a real (non `--fast`)
training run once that phase's scripts exist.

---

## Phase 3: Training pipeline - in progress

**Plan:**
- `training/backbones.py`: `build_backbone(name, pretrained, num_classes=1)`
  factory for `resnet50`, `efficientnet_b0`, `convnext_tiny` (timm),
  `vit_small` (timm, chosen over Transformers for consistency with the
  other three and simpler pretrained-weight handling; recorded per spec),
  `dinov2_small` (Transformers, `linear_probe` and `full_finetune` modes).
  Single-logit head (binary classification via `BCEWithLogitsLoss`), not
  a 2-class softmax head, since the backend ultimately wants one
  probability; recorded in `docs/ASSUMPTIONS.md`.
- `training/metrics.py`: `compute_metrics` (accuracy, sensitivity,
  specificity, precision, F1, ROC-AUC, PR-AUC, Brier score),
  `select_threshold` (highest specificity at a target sensitivity, chosen
  on validation only), `bootstrap_auc_ci`.
- `training/lit_model.py`: `LightningModule` wrapping a backbone: AdamW
  with differential LR (head ~1e-4, backbone ~1e-5), frozen backbone for
  the first few epochs then unfrozen, label smoothing, class weights via
  `pos_weight`, cosine LR schedule.
- `training/baselines.py`: HOG (`skimage.feature.hog`) + `sklearn.svm.SVC`
  baseline, same fold structure as the neural models.
- `training/train.py`: orchestrates all backbones plus the baseline over
  folds, holds out part of the training fold as validation (grouped the
  same way), selects the epoch and decision threshold on validation only,
  evaluates once on the test fold, logs to MLflow (local `mlruns/`) and
  W&B (`WANDB_MODE=offline` when no API key is set). Accepts `--config`,
  `--seed`, `--fast`.
- `training/evaluate.py`: comparison table (CSV + Markdown), ROC curves
  and confusion matrices as Plotly figures, saved to `reports/`.
- `configs/train_resnet50.yaml` (and similar) for the Makefile's
  `train-fast` target.
- Tests for `training/metrics.py` and `training/backbones.py` primarily
  (pure, fast, no GPU needed); `training/train.py`'s own tests use a tiny
  synthetic fixture for the code path (mirroring Phase 1's pattern), while
  `make train-fast` itself is verified against a small real subset of the
  HandPD/NewHandPD data already downloaded, not synthetic images.

**Done when:** `make train-fast` runs every backbone and the baseline on
the tiny fixture end to end, produces the comparison table and figures,
and unit tests for the metrics and threshold selection pass.

**Built:**
- `training/backbones.py`: `build_backbone` (resnet50, efficientnet_b0,
  convnext_tiny, vit_small via timm; dinov2_small via Transformers, with
  `linear_probe`/`full_finetune` modes), `SingleLogitClassifier`,
  `Dinov2Classifier`, `count_parameters`, `get_head_parameters`,
  `freeze_backbone_params`, `unfreeze_all_params`, `get_param_groups`.
- `training/metrics.py`: `Metrics`, `compute_metrics`, `select_threshold`
  (favors sensitivity), `bootstrap_auc_ci`.
- `training/lit_model.py`: `SpiralWaveClassifier` (LightningModule):
  differential-LR AdamW, freeze-then-unfreeze, label smoothing, class
  weights, cosine schedule.
- `training/baselines.py`: HOG + SVM (`extract_hog_features`,
  `train_hog_svm`, `predict_proba`).
- `training/train.py`: `split_train_val`, `sample_fast_subset`,
  `train_one_backbone`, `train_baseline_fold`, `run_training`, a CLI
  (`--config`, `--seed`, `--fast`, `--out`); MLflow logging per run.
- `training/evaluate.py`: `build_comparison_table`, `save_comparison_table`
  (CSV + Markdown), `plot_roc_curves`, `plot_confusion_matrices`
  (pooled across folds/seeds, per-model dropdown), `run_evaluation`.
- `configs/train_resnet50.yaml`.
- Tests: `tests/test_backbones.py` (19), `tests/test_metrics.py` (11),
  `tests/test_lit_model.py` (8), `tests/test_baselines.py` (5),
  `tests/test_train.py` (7), `tests/test_evaluate.py` (9): 59 new tests.

**Verified:**
- Installed `pytorch-lightning`, `timm`, `transformers`, `mlflow`,
  `wandb`, `tabulate` into `.venv` (incrementally, one at a time, after a
  combined install hung in pip's resolver for a long time with zero
  packages landed; seen below under "Bugs found").
- `.venv/Scripts/python.exe -m ruff check .` - "All checks passed!"
- `.venv/Scripts/python.exe -m mypy common training tremor backend tests`
  - "Success: no issues found in 30 source files"
- `.venv/Scripts/python.exe -m pytest -q` - 137 passed.
- The real `make train-fast` command
  (`python -m training.train --fast --config configs/train_resnet50.yaml`)
  run against real HandPD/NewHandPD data (not synthetic): all 5 backbones
  plus the HOG+SVM baseline trained and evaluated successfully, writing
  `reports/training_results.json`. Metrics for the random-init,
  1-epoch, ~10-image fast run cluster near chance (ROC-AUC 0.25 to 0.50
  for the neural backbones, 1.00 for the baseline on this particular tiny
  split) as expected for a smoke test, not a real training result.
- `.venv/Scripts/python.exe -m training.evaluate --results
  reports/training_results.json --reports-dir reports` run for real:
  wrote `reports/model_comparison.csv`, `reports/model_comparison.md`,
  `reports/roc_curves.html`, `reports/confusion_matrices.html` (the
  latter two each about 3.6 MB, confirmed non-empty).
- `mlruns/` populated with a real run per model/fold/seed combination
  (params and metrics logged), confirmed by listing its directory tree.

**Bugs found and fixed during verification:**
- A combined `pip install pytorch-lightning timm transformers mlflow
  wandb` hung in dependency resolution for over 10 minutes with nothing
  installed (confirmed still consuming CPU, not deadlocked, via
  `Get-Process`). Killed it and installed each package separately instead;
  every individual install then completed in well under a minute, since
  each resolves against an already-settled environment rather than five
  packages simultaneously.
- The first `make train-fast` run failed with `OSError: No space left on
  device` while saving a Lightning checkpoint, despite the checkpoint's
  own target directory (`E:\parkinson-screen\checkpoints`, on the `E:`
  drive) having over 10 GB free. Root cause: the system default temp
  directory (`C:\Users\...\AppData\Local\Temp`, used for an intermediate
  atomic-save write) sits on a drive with under 2 GB free, a pre-existing
  condition on this machine. Fixed for this run by pointing `TEMP`/`TMP`
  at a folder on `E:`; fixed in the code by writing checkpoints to a
  `tempfile.TemporaryDirectory` deleted immediately after use instead of
  a persistent `checkpoints/` directory, and using
  `save_weights_only=True` to keep them smaller regardless.
- The first working `--fast` run produced `roc_auc=nan` for every model:
  `sample_fast_subset` samples a small subset per class from across the
  *whole* manifest, then the original code filtered to the single
  lowest-numbered fold from the *full dataset's* fold column. With a
  12-image sample scattered across 5 original folds, that single fold's
  test set could easily (and did) end up with zero images of one class,
  making ROC-AUC undefined. Root-caused by re-running the sampling logic
  directly in a standalone script rather than guessing. Fixed by
  re-assigning fresh folds on the sampled subset itself
  (`assign_folds(manifest, n_folds=2, seed=seed)`); verified robust
  (both classes present in the test fold) across 10 different seeds
  before re-running the full pipeline.
- `split_train_val`'s internal validation split used a fixed `n_splits =
  round(1 / val_fraction)` (5, for the default 0.2), which exceeded the
  smallest class's available sample count on `--fast` mode's tiny folds
  and triggered a `StratifiedGroupKFold` warning. Fixed by capping
  `n_splits` at the smallest class's count.

**Open:**
- `--fast` mode's own selected threshold can still land on a degenerate
  value when the internal train/val split (inside a single already-tiny
  fast-mode fold) happens to have very few examples of one class,
  producing sensitivity/specificity values that are not meaningful (seen
  in the final verification run: several models showed
  `sensitivity=0.000` despite reasonable ROC-AUC ranking ability). This
  does not affect correctness (no crash, no NaN propagation) and will not
  occur at the scale of a real 5-fold run on the full ~1264-image
  dataset, so it was left as-is rather than further tuned for a smoke
  test whose numbers are not meant to be meaningful. Real training (5
  folds x 3 seeds x 5 backbones, `pretrained=true`) needs a GPU (Colab or
  Kaggle) and has not been run; per `CLAUDE.md`, that is expected to
  happen outside this environment.

**Assumptions:** see `docs/ASSUMPTIONS.md` and `docs/DECISIONS.md` (Phase
3 sections).

**Next:** Phase 4 (experiments: ablations, learning curve, shortcut
check, robustness, statistical comparison). Needs from you: nothing to
start the code; the real experiments (not just their `--fast` code paths)
will need the GPU training run from Phase 3 to have happened first.

---

## Phase 4: Experiments - done

**Plan:**
- `training/metrics.py`: add back `paired_bootstrap_comparison` (removed
  from Phase 3's scope deliberately; this is where it belongs).
- `training/data.py`: `SpiralWaveDataset` gets an explicit `augment`
  parameter, decoupled from `train` (which still controls fold
  filtering), so the augmentation ablation can build a "train split, no
  augmentation" dataset.
- `training/ablations.py`: pretrained vs scratch; augmentation vs none;
  spiral only vs wave only vs fused (probability averaging and logistic
  regression stacking, reporting honestly whether fusion helps); full
  fine-tune vs frozen backbone vs DINOv2 linear probe; learning curve at
  25/50/100% of training data. Reuses `training/train.py`'s
  `train_one_backbone` for the actual training loop in every case.
- `training/shortcut_check.py`: erase stroke pixels (threshold, fill with
  local background) and train on background only, compared against
  normal and `skeleton_rerender` preprocessing. Writes the finding
  plainly either way.
- `training/robustness.py`: blur, Gaussian noise, rotation, stroke
  dilation/erosion, JPEG compression at 3 severities, applied to test
  images only; evaluates an already-trained model at each severity level.
- Every script writes tables and Plotly figures to `reports/`, accepts
  `--config`, `--seed`, `--fast`, and reuses the same folds.

**Done when:** every experiment runs in `--fast` mode and produces its
tables and figures, and the code paths are covered by at least smoke
tests.

**Built:**
- `training/metrics.py`: `paired_bootstrap_comparison` (observed AUC
  difference plus a two-sided paired-bootstrap p-value).
- `training/data.py`: `SpiralWaveDataset.augment`, decoupled from `train`.
- `training/train.py`: split `train_one_backbone` into `fit_backbone`
  (train once, pick a threshold on validation) and
  `evaluate_fitted_backbone` (evaluate on any test set, no retraining),
  used directly by `training/robustness.py` to reuse one trained model
  across every corrupted test set. `train_one_backbone` is now a thin
  wrapper over both, unchanged for every existing caller. Also added
  `preprocess_mode`, `augment`, `freeze_epochs` and `dinov2_mode`
  overrides so the ablations never need a second copy of the training
  loop.
- `training/ablations.py`: `run_pretrained_vs_scratch`,
  `run_augmentation_ablation`, `run_finetune_mode_ablation`,
  `run_kind_fusion_ablation` + `fuse_kind_predictions` (probability
  averaging and a logistic-regression stacker over matched
  spiral/wave subjects), `run_learning_curve`, `write_ablation_report`,
  a CLI.
- `training/shortcut_check.py`: `erase_stroke_pixels` (inpaints over the
  stroke mask rather than flat-filling, so real scanner/paper texture
  elsewhere stays intact), `run_shortcut_check`, `render_finding`
  (states "possible shortcut detected" or "no shortcut detected" based on
  whether the background-only model's AUC confidence interval clears
  chance), a CLI.
- `training/robustness.py`: 6 corruption functions (blur, Gaussian noise,
  rotation, dilate, erode, JPEG compression) at 3 severities, applied to
  test images only; `run_robustness_check` fits one model per fold and
  evaluates it against every corrupted variant; a CLI.
- Tests: `tests/test_ablations.py` (8), `tests/test_shortcut_check.py`
  (9), `tests/test_robustness.py` (11): 28 new tests, against the pure
  data-transformation, summarization and plotting functions using the
  existing synthetic image fixtures, not real training (too slow for a
  test suite; the real training path is verified by actually running
  each script, below).

**Verified:**
- `.venv/Scripts/python.exe -m ruff check .` - "All checks passed!"
- `.venv/Scripts/python.exe -m mypy common training tremor backend tests`
  - "Success: no issues found in 33 source files"
- `.venv/Scripts/python.exe -m pytest -q` - 169 passed.
- `.venv/Scripts/python.exe -m training.ablations --config
  configs/train_resnet50.yaml --fast --seed 42` run for real against real
  HandPD/NewHandPD images: all 5 sub-experiments (pretrained vs scratch,
  augmentation, fine-tune mode, kind fusion, learning curve) completed
  and wrote `reports/ablation_results.json` plus a CSV and HTML figure
  per sub-experiment (9 files total).
- `.venv/Scripts/python.exe -m training.shortcut_check --config
  configs/train_resnet50.yaml --fast --seed 42` run for real: wrote
  `reports/shortcut_check_results.json`,
  `reports/shortcut_check_finding.md` ("no shortcut detected": the
  background-only model's AUC, 0.100, sits below chance, not above it),
  and `reports/shortcut_check.html`.
- `.venv/Scripts/python.exe -m training.robustness --config
  configs/train_resnet50.yaml --fast --seed 42` run for real: all 6
  corruption types at all 3 severities (plus the clean baseline)
  evaluated without crashing, wrote `reports/robustness_results.json`,
  `reports/robustness_summary.csv`, `reports/robustness.html`.

**Bugs found and fixed during verification, not assumed correct from the
spec text alone:**
- The `pretrained_vs_scratch` ablation is the first script in this
  project whose `--fast` path actually needs `pretrained=True` (every
  other `--fast` path uses `pretrained=False` specifically to avoid a
  download). `python -m training.ablations --fast` hung indefinitely
  (confirmed genuinely stuck, not slow, via `Get-Process`: over 7 minutes
  elapsed against about 0.1 seconds of CPU time consumed) on
  downloading resnet50's ImageNet weights through Hugging Face Hub's Xet
  storage. Installing `hf_xet` (now pinned in `requirements-train.txt`)
  fixed it: the same download then completed in about 14 seconds. Caught
  by isolating `build_backbone("resnet50", pretrained=True)` in a
  bounded `timeout` wrapper after the full pipeline hung with no output,
  rather than assuming the symptom (a hang) matched its eventual cause.
- `run_kind_fusion_ablation` crashed inside `fuse_kind_predictions` with
  `ValueError: y_true takes value in {} and pos_label is not specified`:
  merging spiral and wave validation predictions on `subject_id` produced
  zero matched rows in this particular `--fast`-mode tiny subsample and
  fold, and `select_threshold` was called unconditionally on the
  (empty) result. Fixed by checking for both classes being present in
  the merged validation and test sets before computing a threshold or
  metrics, returning NaN AUC values and `fusion_helps_vs_best_single:
  None` instead, which is what actually happened, not a crash and not a
  fabricated number.
- Fixing that surfaced a second, deeper crash in the same ablation:
  `split_train_val` raised `ValueError: Cannot have number of splits
  n_splits=2 greater than the number of samples: n_samples=1` for a
  kind-filtered, fold-filtered training subset with a single row.
  `split_train_val`'s own `max(2, ...)` floor on `n_splits` could exceed
  both the smallest class's count and the total sample count at once.
  Fixed by capping `n_splits` at `min(len(manifest),
  smallest_class_count)` and falling back to an empty validation split
  (everything in train) when even a 2-way split is infeasible, rather
  than crashing.
- That empty-validation fallback then surfaced a third problem, not an
  exception this time but a silent 1.4 GB disk leak: with no
  `ModelCheckpoint` callback passed (since there is no validation loss to
  monitor), PyTorch Lightning's `Trainer` defaults to
  `enable_checkpointing=True` and silently adds its own default
  `ModelCheckpoint`, writing to `./checkpoints` relative to the project
  root rather than the intended `tempfile.TemporaryDirectory`. Caught by
  noticing a `Checkpoint directory ... exists and is not empty` log line
  that should never have appeared, then confirming with `du -sh
  checkpoints/` (1.4 GB across 14 files). Fixed with
  `enable_checkpointing=has_validation` on the `Trainer`, and the leaked
  files were deleted (already gitignored via the existing `*.ckpt`
  pattern, so none of it was ever at risk of being committed).
- `fold` values iterated from `sorted(manifest["fold"].unique())` are
  `numpy.int64`, not a native Python `int`; storing them directly in a
  result dict and then `json.dumps`-ing the results crashed with
  `TypeError: Object of type int64 is not JSON serializable`. This had
  not surfaced before because Phase 3's `--fast` path happened to
  hardcode `folds_to_run = [0]` as a literal Python int rather than
  deriving it from `.unique()`, while every ablation script (and the
  real, non-fast path in `training/train.py`) does derive it that way.
  Fixed at each of the 8 call sites across `training/train.py`,
  `training/ablations.py`, `training/robustness.py` and
  `training/shortcut_check.py` by casting with `int(f) for f in
  manifest["fold"].unique()` before sorting.
- `kind_fusion`'s `wave` condition reported `roc_auc: nan` for every fold
  in the final verification run. Checked directly rather than assumed
  benign: the `--fast` sample (6 images per class, shared across both
  kinds) left the wave-only test fold for fold 0 with zero healthy
  images and the test fold for fold 1 with zero parkinson images, so
  `compute_metrics`'s existing both-classes guard correctly reported NaN
  rather than a fabricated AUC. Not a bug: an honest report of
  insufficient `--fast`-mode data for the most data-hungry ablation
  (it needs both kinds present, per fold, per subject).

**Open:**
- Real experiments (5 folds x 3 seeds, `pretrained=true`, full epochs)
  have not been run; per `CLAUDE.md` this needs a GPU (Colab or Kaggle)
  and is expected to happen outside this environment.
- `kind_fusion`'s `--fast`-mode numbers are frequently NaN for the
  reason above; this is inherent to how little data `--fast` mode uses
  split four ways (kind x fold), not something further code changes can
  fix without more `--fast` sample data, which would slow the smoke test
  down for marginal benefit.

**Assumptions:** see `docs/ASSUMPTIONS.md` (Phase 4 sections).

**Next:** Phase 5 (explainability, export and registry). Needs from you:
nothing to start the code; the real Grad-CAM overlays and ONNX export
benefit from a real trained champion model, which needs the GPU run
above.

## Phase 5: Explainability, export and registry - done

**Plan:**
- `training/explain.py`: Grad-CAM for the timm CNNs/ConvNeXt, a ViT/DINOv2
  reshape transform for the transformer backbones, overlays for correct
  and incorrect predictions (failure cases included), plus two
  quantitative checks: (a) fraction of heatmap mass on stroke pixels
  (compared against the strokes' share of image area, and with a
  border-mass fraction so attention to borders or background is reported
  plainly), (b) a deletion test against a random-order baseline.
- `training/register.py`: refit the champion backbone on all data with a
  grouped internal validation split (threshold chosen there), register it
  in MLflow with alias `champion`, threshold stored as a version tag.
- `training/export_onnx.py`: load the champion from the registry, export
  to ONNX (opset 17, dynamic batch, probability output), PyTorch-vs-ONNX
  parity within 1e-4, optional INT8 dynamic quantization with size,
  latency and AUC-change report, write `backend/models/model_card.json`
  and the threshold next to the model. `--fast` registers a tiny dev
  champion first and labels everything as a development model.
- `configs/export.yaml`; tests for all three modules.

**Done when:** the parity test passes and `backend/models/` contains a
model and its card (a tiny fast-mode model is acceptable for development,
clearly labeled as such in the card).

**Built:**
- `training/explain.py`: Grad-CAM with a per-backbone target layer
  (timm CNN/ConvNeXt/ViT blocks, DINOv2 `norm1`), `vit_reshape_transform`
  (drops the class token, folds patches into their grid; works for 14x14
  and 16x16), predicted-class targets, stroke-mass / area / border-mass
  statistics, a deletion test against a random-order baseline, overlays
  for correct and incorrect predictions, and a generated plain-language
  finding that labels `--fast` runs as development runs.
- `training/register.py`: refit the chosen backbone on all data, register
  in MLflow, alias `champion`, threshold as a version tag;
  `load_champion` fails with instructions when none exists.
- `training/export_onnx.py`: ONNX (opset 17, dynamic batch, probability
  output), PyTorch-vs-ONNX parity check that raises if it fails, INT8
  dynamic quantization with size/latency/AUC-change report, model card,
  threshold file.
- `configs/export.yaml`, `backend/models/{model.onnx, model_card.json,
  threshold.json}` (development model, labeled), `reports/explain*`.
- Tests: `tests/test_explain.py` (31), `tests/test_export_onnx.py` (20),
  `tests/test_register.py` (8).

**Verified:**
- `ruff check .` clean; `mypy common training tremor backend tests` - "no
  issues found in 42 source files"; `pytest -q` - 227 passed.
- `python -m training.export_onnx --config configs/export.yaml --fast`
  run for real (trains a dev champion, registers it, exports, checks
  parity, quantizes): parity max abs difference 5.96e-08 against the 1e-4
  requirement; fp32 16.0 MB / 17 ms per image; INT8 4.3 MB / 142 ms.
- `python -m training.explain --config configs/train_resnet50.yaml --fast`
  run for real: 5 overlays (1 correct, 4 incorrect), summary CSV,
  deletion-curve figure, finding. Viewed an overlay: heatmap sits on the
  spiral. The numbers describe a random-init 1-epoch model and the
  finding says so.
- Grad-CAM verified on all five backbones by tests that run it on each,
  which is the actual check on the ViT/DINOv2 reshape transform.

**Bugs found and fixed during verification:**
- The Phase 0 pin `pytorch-grad-cam==1.5.4` does not exist on PyPI; the
  package is `grad-cam`. Installing it then upgraded numpy to 2.x and
  added a second OpenCV build, breaking the pinned stack. Restored both
  and documented the `--no-deps` install in `requirements-train.txt`.
- DINOv2's HF layer has `norm1`, not `layernorm_before` (found by the
  Grad-CAM-on-every-backbone test, then confirmed by listing the
  submodule names).
- `onnx` failed with "DLL load failed" in the full suite but not in
  isolation. Bisected to import order: importing `pandas` or `mlflow`
  before `torch` breaks onnx's DLL on this machine. Fixed in
  `tests/conftest.py` with a comment; `training/export_onnx.py` already
  imports `onnxruntime` first.
- A scripted edit of `explain.py` half-applied (one replacement missed on
  an escaped newline), leaving undefined names; caught immediately by the
  failing tests and fixed by hand.

**Open:**
- INT8 is not a latency win for `efficientnet_b0` on this CPU (about 8x
  slower), only a size win; re-measure for the real champion.
- No real champion exists: the committed model is a development model
  (the card says so) and `cv_metrics` is `TODO(run-on-gpu)`.

**Next:** Phase 6 (LightGBM tremor model and SHAP).


## Phase 6: LightGBM tremor model and SHAP - done

**Built:** `tremor/train_lgbm.py`, `configs/tremor_lgbm.yaml`, `tests/test_train_lgbm.py`,
`backend/models/tremor_lgbm.txt`, `backend/models/tremor_model_card.json`,
`reports/tremor_shap_summary.png`, `reports/tremor_shap_bar.png`.

**Verified:**
- `python -m tremor.train_lgbm --config configs/tremor_lgbm.yaml`: cross-validated AUC 0.998 +/- 0.002 over 5 folds; max |sum(contributions) - raw score| 3.73e-14.
- `pytest tests/test_train_lgbm.py`: 18 passed. Full `pytest -q`: 245 passed.
- `ruff check` clean; `mypy common training tremor backend tests`: no issues in 44 files.

**Findings:**
- No labeled kinematic dataset exists, so the model is trained on simulated spirals only (`trained_on: synthetic`, provenance "Feature pipeline demo, trained on simulated data").
- The AUC is near 1.0 by construction (nuisance parameters are identical for both classes, only the tremor differs) and is not clinical evidence.
- SHAP ranks `spectral_entropy`, `tremor_band_ratio`, `peak_freq_hz` highest and `pen_lifts` at zero. This describes the simulator, not patients.

**Open:** a real labeled kinematic dataset would replace the simulated one; this is a `HANDOFF.md` item.

**Next:** Phase 7 backend.

## Phase 7: Backend (FastAPI) - done

**Built:** `backend/settings.py`, `schemas.py`, `inference.py`, `database.py`, `security.py`,
`main.py`, `tests/test_backend.py` (38 tests). `common/preprocess.py` now imports scikit-image
lazily (only the skeleton mode needs it) so the backend does not need it. `backend/requirements.txt`
gained `opencv-python-headless`. Makefile `backend` target now runs `python -m uvicorn
backend.main:app --reload --port 8000` from the repository root. The frontend client
(`frontend/lib/api.ts`) now saves each live result as a session and reads `{sessions: [...]}`.

**Endpoints (`/api/v1`):** `GET /health`, `POST /predict` (probability, label with an uncertain
margin, threshold, provenance, occlusion map), `POST /tremor` (features, warnings, LightGBM score,
top contributions from `pred_contrib=True`), `POST /sessions`, `GET /sessions?device_id=`,
`DELETE /sessions?device_id=`. Every response, errors included, carries the disclaimer.

**Verified:**
- `pytest tests/test_backend.py`: 38 passed (every endpoint, bad inputs, 503 with no models, size
  limit 413, rate limit 429, CORS, per-device isolation and cap, serving vs training preprocessing
  tensors identical, contributions sum to raw score).
- A real `/predict` call with the 8x8 occlusion map took 0.9 s on CPU (target: under 3 s).
- `ruff check .` clean; `mypy common training tremor backend tests` clean.
- Frontend: `npm run typecheck`, `npm run lint`, `npm test` (9 passed).

**Open:**
- The loaded image model is the labeled development model, so probabilities are not meaningful.
- Tremor analysis latency and peak memory are measured in Phase 9.
- `.gitignore` had a `lib/` rule that silently ignored `frontend/lib/`; it is now anchored (`/lib/`).

**Next:** Phase 9 (Docker, CI, deployment), then Phase 10 docs.

## Phase 8: Frontend (Next.js) - done in development mode

**Plan:**
- Build a Next.js App Router frontend with strict TypeScript, Tailwind CSS, responsive navigation, a calm research-tool visual system, and a persistent screening disclaimer.
- Implement `/`, `/draw`, `/result/[id]`, `/history`, and `/about` without depending on the not-yet-built FastAPI service.
- Add a typed API adapter with a deterministic local mock mode, browser metadata storage, dynamic Plotly charts, and a clear switch to `NEXT_PUBLIC_API_URL`.
- Verify the frontend with typecheck, ESLint, Vitest, a production build, and a route smoke test.

**Built:**
- Next.js 15 App Router scaffold in `frontend/` with `package.json`, `tsconfig.json`, Tailwind, ESLint, Vitest, standalone output, and environment documentation.
- Responsive shared UI in `frontend/components/`: header, footer, disclaimer, status cards, metric cards, occlusion map, Plotly wrapper, delete control, and quality meter.
- Pointer-event drawing canvas in `frontend/components/drawing-canvas.tsx`: spiral/wave templates, coalesced pointer samples, timing, pressure, pointer type, pen-lift markers, touch-action prevention, undo, clear, fixed black-on-white stroke, quality warnings, and PNG capture.
- Pages in `frontend/app/`: landing overview, drawing studio, result review, personal history, and about/model-card pages.
- Typed API and storage modules in `frontend/lib/`: result normalization, mock health/prediction/tremor data, device ID generation, metadata-only browser history, feature formatting, and API session deletion.
- Result review includes probability dial, threshold uncertainty, occlusion map, feature report table, contribution chart, warnings, model version, provenance, and disclaimer.
- History includes screening signal, personal first-session band, tremor-score trend, spiral-deviation trend, session log, and empty state. No population norms or fabricated user history are shown.
- Added Vitest coverage for formatting, local storage, deterministic mock responses, and image-upload validation.
- Added `framer-motion` with reduced-motion support, subtle reveal/stagger primitives, and an animated trace glyph on the landing page.
- Renamed the product to **Kinesis**, removed the brand icon, removed the requested hero label and preview panel, and rebuilt the home page as an asymmetric editorial research interface.
- Added a dedicated image-upload mode to `/draw` with drag-and-drop, keyboard/file selection, PNG/JPEG validation, a 2 MB limit, preview, replace, and analyze actions. It passes the uploaded data URL through the same typed prediction path as live drawing.

**Visual refresh:**
- Reworked the global token layer around warm cream `#faf9f6`, linen, stone, hairline borders, abyssal ink, electric violet, and bioluminescent lime.
- Replaced the soft SaaS treatment with flat editorial surfaces, 4px controls, mono metadata labels, light-weight display type, generous 1200px editorial spreads, and alternating dark/light sections inspired by the supplied Intercom and Integrated Biosciences references.
- Updated the landing hero, navigation, drawing controls, occlusion map, result dial, chart palette, feature table, history cards, about page, and footer to use the new system.
- Removed visible gradients, card shadows, old teal/coral/amber hex values, and oversized rounded-card utilities.

**Latest layout polish (screenshot fix):**
- Widened `.site-container` from 1200px to 1440px so the editorial composition uses the available desktop canvas.
- Gave the hero `calc(100svh - 112px)` height, removed forced four-line display wrapping, reduced display sizes, and added lime/violet underline accents.
- Added viewport-aware minimum heights to the instrument, capture-path, and closing sections, with more generous row spacing and a calmer hover treatment.
- The capture and upload pages remain fully functional while using the same widened shell.

**Verified:**
- `npm install` initially ended with registry `ECONNRESET`; retry with `npm install --prefer-offline --no-audit --no-fund` completed successfully and installed the dependency tree.
- `npm run typecheck` - passed after the build-generated Next type files were available. A parallel typecheck/build attempt failed only because the build cleaned `.next` while `tsc` was reading it; the sequential rerun passed.
- `npm run lint` - passed. Generated `frontend/next-env.d.ts` is excluded from ESLint.
- `npm test` with `TEMP` and `TMP` redirected to `E:\parkinson-screen\.tmp` - 4 test files and 9 tests passed, including invalid-file and valid-image upload cases. The first retry hit the known system-drive `ENOSPC` condition; the workspace-temp retry passed.
- `npm run build` - passed; routes `/`, `/about`, `/draw`, `/history`, and dynamic `/result/[id]` were generated successfully after the Kinesis redesign and upload flow.
- Standalone server route smoke test on port 3106 - `/`, `/draw`, `/draw?mode=upload`, `/history`, `/about`, and `/result/demo-smoke` each returned HTTP 200.
- Latest screenshot-fix pass: `npm run typecheck`, `npm run lint`, and `npm test` (4 files, 9 tests) passed. The standard production build compiled the app but its build worker then hit a Windows out-of-memory abort because an already-running Next dev server was using about 1.3 GB. The live dev routes returned 500 while that process was memory-constrained. Stop and restart the dev server, then rerun `npm run build` and the route smoke test.

**Open:**
- Phases 5 to 7 are not built, so the frontend currently uses a clearly labelled deterministic development adapter. Set `NEXT_PUBLIC_API_URL` and `NEXT_PUBLIC_USE_MOCK_API=false` after the backend contract is available.
- A real browser walkthrough of pointer capture, image upload, Framer Motion transitions, and Plotly interaction is still needed on a touchscreen, stylus, mouse, and mobile viewport.
- `npm start` prints a Next warning because the project uses standalone output; Phase 9 deployment should launch the generated standalone server or remove standalone output for the chosen hosting target.

**Assumptions:**
- The frontend uses metadata-only browser persistence. Raw images and raw stroke points are not persisted by the frontend; the intended backend will store features and scores only.
- Mock result values are visibly marked as a development preview and must not be reported as model metrics.
- HandPD and NewHandPD are the planned research training sources; the UI makes no claim about browser data being part of those datasets.

**Next:** Phase 9 deployment work can use this frontend, but Phase 7 backend integration should be completed first if live model results are required. Needs from you: nothing to start the frontend code; manual browser testing and the Phase 7 API contract are still needed for a fully connected demo.

## Phase 9: Docker, CI and deployment - written, Docker builds not verified here

**Built:** `backend/Dockerfile` (multi-stage, `python:3.11-slim`, non-root user, runtime packages
only, health check), `frontend/Dockerfile` (multi-stage, Next.js standalone, non-root, health
check), `.dockerignore`, `frontend/.dockerignore`, `docker-compose.yml` (frontend, backend,
Postgres with a volume), `render.yaml` (two web services plus managed Postgres),
`.github/workflows/ci.yml` (python, backend-only, frontend, docker build jobs).

**Verified (commands run on this machine):**
- A clean venv built from only `backend/requirements.txt` (no torch, no shap, no scikit-learn,
  pydantic 2.8.2 as pinned): `pytest tests/test_backend.py` gave 39 passed. This proves the
  backend runs without the training stack.
- A real `uvicorn backend.main:app` server answered `/api/v1/health` and `/api/v1/tremor`.
- `docker compose config -q` accepts `docker-compose.yml`.
- `npm run build` with `NEXT_PUBLIC_USE_MOCK_API=false` produced the standalone output;
  `typecheck`, `lint` and `npm test` pass.
- `ruff check .` clean, `mypy` clean, full `pytest -q`: 284 passed.

**Measured (backend-only venv, Windows, CPU, development image model):**
- `/predict` with the 8x8 occlusion map (65 forward passes): median 1.3 s, max 1.4 s (target: under 3 s).
- `/tremor` on 960 points: median 22 ms, max 87 ms (target: under 300 ms).
- Peak working set for the whole process: 298 MB. Before the fix it was 2.0 GB, because one batch of
  65 images allocated huge activations. Inference now runs in chunks of 8 with the ONNX memory
  arena off, which is what makes the 512 MB Render limit realistic.
- Installed packages in that venv: about 550 MB, so the image is expected to be roughly 700 MB.
  This is an estimate, not a measured image size.

**Not verified (open):**
- No Docker image was built: the Docker daemon is not running on this machine, and the system
  drive has under 2 GB free. `docker compose up` and the `docker` CI job are therefore untested.
  Run `docker compose up --build` once (listed in `HANDOFF.md`) and record the image size with
  `docker images`.
- The CI workflow has never run (nothing is pushed). Its dependency install order is untested on
  Linux.
- Render was not deployed (needs the user's account).
- Peak memory inside a 512 MB container on Linux may differ from the Windows working set.

**Next:** Phase 10 documentation.

## Phase 10: Documentation and report assets - done

**Built:** `README.md` (overview, architecture diagram, quick start, make targets, reproduction,
deployment, limitations), `docs/MODEL_CARD.md`, `docs/VIVA_QA.md` (20 questions), `reports/REPORT_ASSETS.md`,
`HANDOFF.md`, and `docs/IDEAS.md`. `docs/DATASHEET.md` is the Phase 1 audit output, unchanged.

**Verified:** every command in the README reproduction list matches a script that exists and
accepts `--config`; asset paths in `reports/REPORT_ASSETS.md` were checked against `reports/`;
no long dashes in any document; the model card repeats only numbers that exist in committed
files (parity 6e-8, threshold 0.5233, simulated AUC 0.998).

**Open:** every image-model metric is still `TODO(run-on-gpu)`. `HANDOFF.md` lists the manual steps.
