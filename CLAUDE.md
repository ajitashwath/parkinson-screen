# Parkinson's Disease Screening from Spiral and Wave Drawings Using Deep Learning

## 0. Role and objective

You are a senior ML engineer and full-stack developer. Build the project "Parkinson's Disease Screening from Spiral and Wave Drawings Using Deep Learning" end to end, in this repository, following the methodology below. The project is for a 4th-year B.Tech AI/ML final-year evaluation. It must be technically solid, honest about its limits, reproducible, and deployable. It must look deep because it is rigorous, not because it is complicated.

## 1. Non-negotiable rules

1. **Never fabricate results.** Do not write metrics, tables or plots that were not produced by code you ran. If a full training run is impossible in your environment, build and verify the pipeline with a `--fast` smoke run, and leave the report tables to be filled by real runs. Mark any placeholder as `TODO(run-on-gpu)`.
2. **Never fabricate data or dataset facts.** If a dataset is not present, write clear download instructions in `data/README.md` and make the code fail with a helpful message. Do not invent file names, subject IDs or labels.
3. **Honesty about scope.** The tool is a "screening aid, not a diagnosis". This disclaimer must appear in the UI on every result screen, the README, the model card and the API responses. Never use the words "diagnose" or "detect Parkinson's" in UI copy. Use "screening" and "possible signs".
4. **Verify before claiming done.** After every task, run the relevant tests or commands and check the output. Do not say something works unless you ran it. Report failures plainly.
5. **No scope creep.** Build what is specified. If you think something extra is valuable, add it to `docs/IDEAS.md` instead of building it.
6. **No secrets in the repo.** Use environment variables and `.env.example`.
7. **Keep it simple.** Prefer plain, readable code and standard libraries over clever abstractions. No em dashes in any docs or UI text. Never name a function starting with an underscore.

## 2. Tech stack (fixed)

**Training:** Python 3.11, PyTorch, PyTorch Lightning, timm, Hugging Face Transformers, scikit-learn, scikit-image, OpenCV, pytorch-grad-cam, Weights and Biases (optional at runtime), MLflow, Plotly, pandas, PyYAML.

**Tremor analysis:** NumPy, SciPy, LightGBM, SHAP (offline only), pytest.

**Backend:** FastAPI, Pydantic v2, onnxruntime (CPU), NumPy, SciPy, LightGBM, SQLAlchemy, psycopg (Postgres), uvicorn. No PyTorch, no `shap` package in the backend image.

**Frontend:** Next.js (App Router), TypeScript, Tailwind CSS, `react-plotly.js` (dynamic import, no SSR).

**Infra:** Docker, docker-compose, Render (`render.yaml`), GitHub Actions, Makefile, ruff, mypy (light), ESLint.

Keep two separate dependency sets: `requirements-train.txt` and `backend/requirements.txt`. Training libraries must never enter the backend image.

## 3. Repository layout

```
parkinsons-screening/
├── CLAUDE.md                  # this prompt
├── PROGRESS.md                # phase checklist, updated after every phase
├── docs/                      # ASSUMPTIONS.md, DECISIONS.md, IDEAS.md, MODEL_CARD.md, DATASHEET.md, VIVA_QA.md
├── configs/                   # YAML: data.yaml, train_*.yaml, export.yaml
├── data/                      # README.md (instructions), raw/ (gitignored), processed/, folds.csv
├── common/
│   └── preprocess.py          # THE shared image preprocessing (used by training AND backend)
├── training/
│   ├── data.py                # dataset, augmentations, grouped stratified k-fold
│   ├── backbones.py           # factory: timm (resnet50, efficientnet_b0, convnext_tiny) + HF (vit, dinov2-small)
│   ├── lit_model.py           # LightningModule
│   ├── baselines.py           # HOG + SVM
│   ├── train.py               # run all models over folds, log to W&B + MLflow
│   ├── ablations.py           # pretrained vs scratch, aug vs none, spiral/wave/fused, freeze vs full FT
│   ├── shortcut_check.py      # stroke-masked (background-only) test
│   ├── robustness.py          # corruption suite
│   ├── evaluate.py            # metrics, ROC, PR, confusion matrices, Plotly figures
│   ├── explain.py             # Grad-CAM (offline) + faithfulness sanity checks
│   ├── export_onnx.py         # torch -> ONNX, parity check, optional INT8
│   └── register.py            # MLflow registry + "champion" alias
├── tremor/
│   ├── features.py            # resample, smooth, features
│   ├── synthetic.py           # spiral generator with known tremor
│   ├── validate.py            # recovery plots (injected vs measured frequency, dose-response)
│   └── train_lgbm.py          # LightGBM + offline SHAP plots
├── backend/                   # FastAPI app, Dockerfile, requirements.txt, models/ (onnx, lgbm, model_card.json)
├── frontend/                  # Next.js app, Dockerfile
├── tests/                     # pytest for common, tremor, backend
├── reports/                   # generated tables and Plotly HTML (from real runs only)
├── .github/workflows/ci.yml
├── docker-compose.yml
├── render.yaml
├── Makefile
├── requirements-train.txt
└── README.md
```

## 4. Methodology: how to work

Work in the phases below, **strictly in order**. For each phase:

1. Write a short plan in `PROGRESS.md` (what files, what tests, what "done" means).
2. Implement in small steps. Write tests alongside the code, not after.
3. Run `make check` (ruff, mypy, pytest). Fix everything before moving on.
4. Update `PROGRESS.md` with what was done, what was verified (with the command run), and any open issue.
5. Make one git commit per phase with a clear message (`phase-3: lightning training pipeline and baselines`).
6. **Stop at the end of the phase and wait for my go-ahead** unless I said "continue".

If something is ambiguous, pick the most reasonable default, record it in `docs/ASSUMPTIONS.md`, and continue. Only ask me a question if you are truly blocked.

Every script must accept `--config`, `--seed`, and `--fast` (tiny subset, 1 epoch, CPU) and must be runnable from the Makefile. Set all seeds (Python, NumPy, torch) and record them in the logs. Logging to W&B must be optional: if no API key is present, use `WANDB_MODE=offline`. MLflow uses a local `mlruns/` store by default.

---

## Phase 0: Scaffold

- Create the repo layout, `Makefile` targets (`setup`, `check`, `test`, `train-fast`, `export`, `backend`, `frontend`, `up`), `.gitignore`, `.env.example`, ruff and mypy config, `PROGRESS.md`, `docs/` stubs.
- Pin dependencies in both requirements files.
- **Done when:** `make setup && make check` passes on an empty codebase with one placeholder test.

## Phase 1: Data layer and preprocessing

1. **Dataset audit first.** Write `data/README.md` describing where to get: Kaggle "Parkinson's Drawings" (spiral and wave, healthy vs Parkinson's), HandPD/NewHandPD, and optionally PaHaW. Then write `training/data.py` to read whatever layout is present (`data/raw/<dataset>/<spiral|wave>/<healthy|parkinson>/*.png`).
2. Write an audit script that prints: image counts per class and drawing type, image sizes, duplicates (perceptual hash), and **whether subject IDs are available**. Save the summary to `docs/DATASHEET.md`.
3. **Leakage control:** if subject IDs exist, use `StratifiedGroupKFold` by subject. If they do not exist (some Kaggle sets), fall back to stratified k-fold plus duplicate grouping, and **document the limitation loudly** in the datasheet and report. Augmentation happens only on training folds, on the fly, never saved to disk before splitting.
4. **`common/preprocess.py` is the single source of truth** for image preprocessing, used by training and the backend: grayscale, crop to the drawing's bounding box with padding, resize to a square keeping aspect ratio (white padding), normalize, replicate to 3 channels. Optional mode `skeleton_rerender` (skeletonize, then draw at fixed stroke width) for the shortcut and robustness experiments. Preprocessing must be deterministic and unit tested.
5. Augmentations (train only): small rotation, slight scale and translation, brightness/contrast, random erasing. No strong elastic warps (they distort tremor patterns).
6. Save the fold assignment in `data/folds.csv` for reproducibility.

**Done when:** the audit runs on a tiny synthetic fixture (generate a few fake spiral images in tests), folds have no subject or duplicate overlap (tested), and preprocessing tests pass.

## Phase 2: Tremor analysis package

Build `tremor/features.py`. Input: a list of points `(x, y, t_ms, pressure, pointer_type)` plus optional pen-lift markers.

Steps:
1. Validate input (minimum duration 2 s, minimum 50 points, effective sampling rate). Return warnings instead of crashing (for example, "sampling rate too low", "mouse input adds noise").
2. Resample to a uniform 100 Hz with interpolation.
3. Smooth with Savitzky-Golay, then compute speed, acceleration and jerk.
4. Compute these features (plain names, documented in a docstring):
   - `duration_s`, `path_length`, `mean_speed`, `speed_cv` (speed variation)
   - `jerk_rms` (normalized by duration and path length)
   - `pen_lifts`, `pause_ratio`
   - `tremor_band_ratio`: power in 4 to 12 Hz divided by total power, from a Welch spectrum of the speed signal
   - `peak_freq_hz`: dominant frequency in the 3 to 12 Hz band
   - `spectral_entropy`
   - `spiral_deviation`: fit an Archimedean spiral (`r = a + b*theta`) to the drawing, report residual RMS divided by mean radius
   - `pressure_mean`, `pressure_sd` (only if the pointer type is pen)

Build `tremor/synthetic.py`: generate an ideal spiral, add a band-limited oscillation at a chosen frequency (3 to 12 Hz) and amplitude, plus small noise, with configurable sampling jitter and a "mouse noise" mode.

Build `tremor/validate.py` (this is the piece that impresses evaluators):
- Plot **injected frequency vs recovered `peak_freq_hz`** (should follow the diagonal).
- Plot **tremor amplitude vs `tremor_band_ratio`** (should increase steadily).
- Plot feature stability under sampling jitter and mouse-level noise.
- Save Plotly HTML figures to `reports/` (generated by running the code).

**Tests (pytest, using the synthetic generator as ground truth):**
- Recovered frequency within 0.5 Hz of the injected frequency across the 3 to 12 Hz range.
- `tremor_band_ratio` increases with amplitude.
- Too-short or too-sparse input returns warnings, not exceptions.
- Features are deterministic for the same input.

**Done when:** all tremor tests pass and the validation figures are generated.

## Phase 3: Training pipeline

1. `training/backbones.py`: factory returning a model with a 2-class (or single-logit) head. Backbones: `resnet50`, `efficientnet_b0`, `convnext_tiny` (timm), `vit_small` (Transformers or timm, whichever loads reliably, record which), `dinov2_small` (Transformers, supports linear-probe and full fine-tune modes). All ImageNet or self-supervised pretrained by default, with a `pretrained=false` option for ablations.
2. `training/lit_model.py`: LightningModule with AdamW, differential learning rates (head about 1e-4, backbone about 1e-5), a few epochs with a frozen backbone then unfreeze, label smoothing, class weights, cosine schedule, early stopping on validation loss, checkpoint the best epoch.
3. `training/baselines.py`: HOG + SVM (scikit-learn), trained with the same folds.
4. `training/train.py`: for each model and each of 5 folds (and 3 seeds when not `--fast`): hold out part of the training fold as validation (grouped the same way), pick the epoch **and the decision threshold** on validation only (threshold chosen to favor sensitivity, for example the highest specificity at a target sensitivity), then evaluate once on the test fold. **Never tune anything on the test fold.**
5. Metrics per fold: accuracy, sensitivity, specificity, precision, F1, ROC-AUC, PR-AUC, Brier score. Also parameter count and CPU inference latency. Aggregate as mean plus or minus std across folds and seeds. Report a bootstrap confidence interval for AUC.
6. Log to W&B (curves, per-fold metrics) and MLflow (params, metrics, artifacts).
7. `training/evaluate.py`: comparison table (CSV plus Markdown), ROC curves, confusion matrices, all as Plotly figures saved to `reports/`.

**Done when:** `make train-fast` runs every backbone and the baseline on the tiny fixture end to end, produces the comparison table and figures, and unit tests for the metrics and threshold selection pass.

## Phase 4: Experiments (the depth)

Implement each as a script that reads the same folds and writes tables and Plotly figures to `reports/`:

1. **Ablations** (`ablations.py`): pretrained vs scratch; augmentation vs none; spiral only vs wave only vs both fused (average probabilities, also try logistic-regression stacking and report honestly whether fusion actually helps); full fine-tune vs frozen backbone vs DINOv2 linear probe.
2. **Learning curve:** train with 25%, 50%, 100% of the training data. Shows which model family is most data efficient.
3. **Shortcut check** (`shortcut_check.py`): erase the stroke pixels (threshold, then fill with the local background) and train on background only. If performance is clearly above chance, the dataset leaks paper or scanner cues. Also compare against the `skeleton_rerender` preprocessing. Write the finding plainly, whatever it is.
4. **Robustness** (`robustness.py`): apply blur, Gaussian noise, rotation, stroke thickness change (dilate and erode) and JPEG compression at 3 severity levels to the **test images only**. Plot AUC vs severity per model.
5. **Statistical honesty:** with small n, many differences will not be significant. Add a paired bootstrap comparison between the top two models and state clearly when the difference is within noise.

**Done when:** every experiment runs in `--fast` mode and produces its tables and figures, and the code paths are covered by at least smoke tests.

## Phase 5: Explainability, export and registry

1. `training/explain.py`: Grad-CAM for CNNs and ConvNeXt, and a correct reshape transform for ViT/DINOv2. Produce overlays for correct and incorrect predictions (including failure cases). Add two simple quantitative checks: (a) the fraction of heatmap mass that lies on the stroke pixels, (b) a deletion test (remove the most important regions, see how quickly the probability falls). Note honestly if the model attends to borders or background.
2. `training/register.py`: register the chosen model in MLflow with the alias `champion`.
3. `training/export_onnx.py`: export the champion to ONNX (opset 17, dynamic batch). **Parity test:** PyTorch and ONNX outputs on a test batch must match within 1e-4. Optionally apply INT8 dynamic quantization and report file size, latency and the AUC change. Write `backend/models/model_card.json` (model name, version, input spec, threshold, training data, CV metrics, known limits).
4. Save the chosen decision threshold next to the model.

**Done when:** the parity test passes, `backend/models/` contains a model and its card (a tiny fast-mode model is acceptable for development, clearly labeled as such in the card).

## Phase 6: LightGBM tremor model and SHAP

1. `tremor/train_lgbm.py`: if a labeled kinematic dataset is present in `data/`, train and cross-validate LightGBM on the tremor features with grouped folds. If not, train **only on synthetic spirals** (tremor vs no tremor) and set `"trained_on": "synthetic"` in the model card. The backend and UI must display this provenance ("Feature pipeline demo, trained on simulated data") whenever the model is synthetic.
2. Offline: generate SHAP summary and bar plots with the `shap` library and save them as figures.
3. Save the LightGBM booster in text format for the backend. The backend will use `predict(..., pred_contrib=True)` for per-feature contributions.

**Done when:** the model and card are saved, and a test checks that contributions sum to the raw prediction.

## Phase 7: Backend (FastAPI)

- Endpoints under `/api/v1`:
  - `GET /health` returns status and model versions.
  - `POST /predict`: body has a base64 PNG of the drawing and `kind` (`spiral` or `wave`). Returns probability, screening label (`lower_concern` / `higher_concern` / `uncertain`), threshold used, model version, provenance, and an **occlusion sensitivity map** (slide a patch over the preprocessed image with onnxruntime and measure the probability drop; grid size configurable, default 8x8, cap the total forward passes). Label the map "occlusion map" in the response, not "Grad-CAM". The `uncertain` label applies when the probability is within a configurable margin of the threshold.
  - `POST /tremor`: body has points and pointer type. Returns features, warnings, the LightGBM output if a model is loaded, top contributing features, and provenance.
  - `POST /sessions`, `GET /sessions?device_id=...`, `DELETE /sessions?device_id=...` (delete all my data).
- Storage: SQLAlchemy models; SQLite locally, Postgres via `DATABASE_URL`. Store **features and scores only**. Do not store raw drawings or raw strokes.
- The backend must call `common/preprocess.py` (copy it into the image at build time) so training and serving preprocess identically. Add a test that proves the same input gives the same tensor in both paths.
- Validation and safety: Pydantic limits (max 20,000 points, max image 2 MB, valid ranges), restricted CORS from `ALLOWED_ORIGINS`, simple rate limiting, request size limit, consistent error format, structured logging. Every response includes the disclaimer field.
- Load models once at startup. If model files are missing, `/health` reports it and the prediction endpoints return a clear 503.

**Done when:** `TestClient` tests cover every endpoint including bad inputs, and the app runs with `make backend`.

## Phase 8: Frontend (Next.js)

Pages:
- `/` landing page: what it is, how it works, the disclaimer.
- `/draw`: choose spiral or wave, show a faint template to trace, capture with **pointer events** (`getCoalescedEvents` for higher sampling, `touch-action: none`, prevent page scroll while drawing, record x, y, `performance.now()`, pressure, pointer type). Fixed stroke width, black on white, so canvas images match the training images. Buttons: clear, undo, submit. Show a hint to use a touchscreen or stylus for better data.
- `/result/[id]`: model probability and label, the occlusion map overlay, the tremor report (feature table with plain-language explanations), the SHAP-style contribution bar chart (Plotly), warnings, provenance badge, and the persistent disclaimer.
- `/history`: Plotly line charts of tremor score and spiral deviation over sessions, with the user's own baseline band (mean of first sessions). No population norms.
- `/about`: model card summary, dataset, known limitations, and a "delete my data" button.

Quality: TypeScript strict, typed API client, loading and error states, responsive layout, accessible labels and contrast, `NEXT_PUBLIC_API_URL` env var, device ID generated once and kept in `localStorage`.

**Done when:** `npm run build`, lint and type-check pass, and a manual walkthrough (or a Playwright smoke test if easy) works: draw, submit, see result, see history.

## Phase 9: Docker, CI and deployment

- `backend/Dockerfile`: multi-stage, `python:3.11-slim`, non-root user, CPU-only dependencies, health check.
- `frontend/Dockerfile`: multi-stage, Next.js standalone output.
- `docker-compose.yml`: frontend, backend, Postgres with volumes and env vars. `make up` should bring everything up.
- `render.yaml`: two web services plus a managed Postgres. Note the free-tier limits (512 MB RAM, sleeps when idle, no persistent disk). If the model files exceed GitHub's 100 MB limit, use INT8 or Git LFS, and document it.
- `.github/workflows/ci.yml`: ruff, mypy, pytest, frontend lint and build on every push.
- Measure and report the backend image size and peak memory during a prediction.

**Done when:** `docker compose up` works locally end to end and CI passes.

## Phase 10: Documentation and report assets

- `README.md`: overview, architecture diagram, quick start, `make` targets, how to reproduce results, deployment steps, limitations.
- `docs/MODEL_CARD.md`: intended use, out-of-scope use, data, metrics (from real runs), limits, ethical notes.
- `docs/DATASHEET.md`: from the Phase 1 audit.
- `docs/VIVA_QA.md`: 20 likely examiner questions with short honest answers based on this project's real design (leakage handling, why sensitivity, why ONNX, why occlusion vs Grad-CAM, why LightGBM is descriptive or synthetic, what the shortcut check found, domain shift between scans and canvas drawings, why no clinical claims).
- `reports/REPORT_ASSETS.md`: an index of every table and figure the final report needs, and which command generates each.
- A final `HANDOFF.md` listing exactly what I must still do manually (datasets, GPU runs, API keys, Render deploy, volunteer test-retest).

---

## 5. Engineering standards

- Type hints on public functions, short docstrings, small modules, no dead code.
- Config in YAML, not hard-coded. No magic numbers without a named constant.
- Tests: unit tests for preprocessing, metrics, threshold selection, tremor features and the synthetic generator; API tests; ONNX parity; feature contribution sum check. Target: every module in `common/`, `tremor/` and `backend/` has tests.
- Deterministic where possible; document any nondeterminism.
- Git: one commit per phase, meaningful messages, no large binaries except the small model files noted above.
- Error handling: fail loudly in research code (clear messages), fail gracefully in the API (structured errors).
- Performance target for the backend on CPU: prediction under about 3 seconds including the occlusion map, tremor analysis under 300 ms.

## 6. Known risks to handle explicitly

| Risk | Handling |
|---|---|
| Tiny dataset, unstable metrics | Repeated grouped k-fold, mean plus or minus std, bootstrap CI, no over-claiming |
| No subject IDs in a dataset | Document the leakage risk, group by duplicates, prefer datasets with IDs |
| Domain shift (paper scans vs canvas) | Shared preprocessing, skeleton re-render option, state as a limitation |
| Model learns paper or scanner cues | Shortcut check, report result honestly |
| Mouse input is noisy | Warn in UI, recommend touch or stylus |
| No labeled stroke data | LightGBM labeled as synthetic, tremor panel described as descriptive |
| Render free tier memory | ONNX only, INT8 option, small model, no torch or `shap` in backend |
| Cold start on Render | Document, tell the user to open the demo a minute early |

## 7. Response format after each phase

End every phase with a short report:
1. What you built (files).
2. Exactly which commands you ran to verify it, and their outcome.
3. Anything that failed or is still open.
4. Assumptions you made.
5. The next phase, and what you need from me (if anything).

Begin with **Phase 0**. Do not start Phase 1 until I reply.
