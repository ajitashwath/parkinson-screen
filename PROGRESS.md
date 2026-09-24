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

## Phase 4: Experiments - in progress

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

## Phase 5: Explainability, export and registry - not started

## Phase 6: LightGBM tremor model and SHAP - not started

## Phase 7: Backend (FastAPI) - not started

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

## Phase 9: Docker, CI and deployment - not started

## Phase 10: Documentation and report assets - not started
