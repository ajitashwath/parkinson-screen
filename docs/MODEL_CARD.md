# Model card

**Screening aid, not a diagnosis.** Possible signs only. Not clinically validated.

This card covers two models served by the backend. Machine-readable versions are
`backend/models/model_card.json` and `backend/models/tremor_model_card.json`.

## Status in one paragraph

No real cross-validation run has been done yet, so this card contains **no accuracy figures for the
image model**. Every metric is marked `TODO(run-on-gpu)`. The image model currently shipped is a
development model trained briefly on CPU to prove the pipeline works. The tremor model is trained on
simulated data.

## 1. Image model (spiral and wave drawings)

| | |
|---|---|
| Name | `parkinsons-screening-champion`, version 1 |
| Shipped weights | **Development model** (`development_model: true`): `efficientnet_b0` as a placeholder, refit on all images for a short run. Not a real screening model. |
| Format | ONNX opset 17, dynamic batch, 16 MB |
| Input | `image`, float32, `[batch, 3, 224, 224]`, values in [0, 1], from `common/preprocess.py` (standard mode), channels first. No ImageNet mean and std. |
| Output | `probability`, float32, `[batch]`, sigmoid already applied. Probability of the Parkinson's class. |
| Decision threshold | 0.5233, chosen on a grouped validation split as the highest specificity at a target sensitivity of 0.9. Chosen on validation only, never on test folds. |
| Labels returned | `higher_concern`, `lower_concern`, or `uncertain` when within 0.05 of the threshold. |
| Parity | PyTorch and ONNX outputs match to 6e-8 on 16 images (contract: 1e-4). |

### Intended use

Demonstrating how a screening pipeline for drawing-based tremor and shape features can be built,
validated honestly, and deployed. Coursework and research exploration.

### Out of scope

Diagnosis, monitoring a diagnosed patient, treatment decisions, insurance or employment decisions,
use on children, or any use where a person might rely on the result instead of seeing a clinician.
The "lower concern" label is not reassurance.

### Training data

HandPD and NewHandPD (UNESP): 1264 scanned images, 158 subjects, spiral and meander (wave)
drawings, healthy versus Parkinson's. Folds are grouped by subject so no person appears in both
training and test. Details and counts: `docs/DATASHEET.md`. Augmentation is applied on the fly to
training folds only (small rotation, scale, translation, brightness, contrast, random erasing; no
elastic warps).

### Metrics

| Metric | Value |
|---|---|
| Cross-validated AUC, sensitivity, specificity, F1, PR-AUC, Brier score | `TODO(run-on-gpu)` |
| Bootstrap 95% CI for AUC | `TODO(run-on-gpu)` |
| Paired bootstrap between the top two models | `TODO(run-on-gpu)` |

The `--fast` smoke run (`reports/model_comparison.md`) trained on a handful of images for one epoch.
Those numbers prove the code runs and say nothing about model quality; do not quote them.

### Known limits

- Trained on paper scans. Canvas, touchscreen and mouse drawings are a different domain and have not
  been evaluated.
- Small data: 158 subjects, so estimates will be wide.
- A model can learn scanner or paper cues. The shortcut check exists to test this; its current result
  is from a `--fast` run and is not evidence.
- The occlusion map shows where covering the image changes the score. It is not a causal explanation.
- Not calibrated for any population. Not tested across ages, devices, handedness or other conditions
  that affect drawing (essential tremor, arthritis, stroke, medication).

## 2. Tremor feature model (LightGBM)

| | |
|---|---|
| Name | `tremor-feature-lightgbm` |
| Trained on | **Simulated spirals only** (`trained_on: synthetic`). Shown in the UI as "Feature pipeline demo, trained on simulated data". |
| Label | 1 means a tremor was injected into the simulated spiral, 0 means none. |
| Features | 13: duration, path length, mean speed, speed variation, jerk, pen lifts, pause ratio, tremor band ratio (4 to 12 Hz), peak frequency, spectral entropy, spiral deviation, pen pressure mean and spread. |
| Contributions | LightGBM `pred_contrib=True`, exact TreeSHAP, summing to the raw score (largest gap measured 3.7e-14). |
| Cross-validated AUC | 0.998 +/- 0.002 over 5 folds, on simulated data. This is close to 1 by construction and is not evidence of clinical accuracy. |

The tremor feature code itself was validated against simulated tremors with known frequency and
amplitude: the recovered peak frequency stays within 0.5 Hz across 3 to 12 Hz, and the tremor band
ratio rises with amplitude (`reports/tremor_*.html`, tests in `tests/test_tremor_features.py`).

SHAP importances (`reports/tremor_shap_*.png`) rank spectral entropy, tremor band ratio and peak
frequency highest and pen lifts at zero. That describes the simulator, not patients.

## Ethical notes

- The training data are published research datasets; this project did not collect them and cannot
  verify their consent terms beyond what the dataset page states. No personal data of app users
  is stored beyond anonymous feature values and scores tied to a random device ID. Raw drawings and
  raw strokes are never stored. Users can delete their data from the About page.
- A false "lower concern" could delay someone seeking care; a false "higher concern" could cause
  distress. This is why the threshold favors sensitivity, an uncertain band exists, and every screen
  carries the disclaimer.
- The wording avoids "diagnose" and "detect Parkinson's".
