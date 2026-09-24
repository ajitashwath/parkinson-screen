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

## Phase 3: Training pipeline - not started

## Phase 4: Experiments - not started

## Phase 5: Explainability, export and registry - not started

## Phase 6: LightGBM tremor model and SHAP - not started

## Phase 7: Backend (FastAPI) - not started

## Phase 8: Frontend (Next.js) - not started

## Phase 9: Docker, CI and deployment - not started

## Phase 10: Documentation and report assets - not started
