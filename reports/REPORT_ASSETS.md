# Report assets

Every table and figure the final report needs, and the command that produces it. Commands run from
the repository root. "Needs GPU run" means the file in `reports/` today comes from a `--fast` smoke
run (a handful of images, one epoch) and must be regenerated without `--fast` before it is quoted.
Nothing here is handwritten.

| Asset | File | Command | Status |
|---|---|---|---|
| Dataset counts, sizes, duplicates, subject IDs | `docs/DATASHEET.md` | `python -m training.audit_data --config configs/data.yaml` | Real (full dataset) |
| Fold assignment | `data/folds.csv` | same command | Real |
| Tremor: injected vs recovered frequency | `reports/tremor_frequency_recovery.html` | `python -m tremor.validate --config configs/tremor.yaml` | Real |
| Tremor: amplitude vs tremor band ratio | `reports/tremor_amplitude_response.html` | same command | Real |
| Tremor: stability under jitter and mouse noise | `reports/tremor_stability.html` | same command | Real |
| Model comparison table | `reports/model_comparison.csv`, `.md` | `python -m training.train --config configs/train_resnet50.yaml` | Needs GPU run |
| Raw per-fold results | `reports/training_results.json` | same command | Needs GPU run |
| ROC curves | `reports/roc_curves.html` | same command | Needs GPU run |
| Confusion matrices | `reports/confusion_matrices.html` | same command | Needs GPU run |
| Ablation: pretrained vs scratch | `reports/ablation_pretrained_vs_scratch.*` | `python -m training.ablations --config configs/train_resnet50.yaml` | Needs GPU run |
| Ablation: augmentation vs none | `reports/ablation_augmentation.*` | same command | Needs GPU run |
| Ablation: spiral, wave, fused | `reports/ablation_kind_fusion.*` | same command | Needs GPU run |
| Ablation: full fine-tune, frozen, linear probe | `reports/ablation_finetune_mode.*` | same command | Needs GPU run |
| Learning curve (25, 50, 100 percent) | `reports/ablation_learning_curve.*` | same command | Needs GPU run |
| Shortcut check table and finding | `reports/shortcut_check_results.json`, `.html`, `shortcut_check_finding.md` | `python -m training.shortcut_check --config configs/train_resnet50.yaml` | Needs GPU run |
| Robustness: AUC vs corruption severity | `reports/robustness.html`, `robustness_summary.csv` | `python -m training.robustness --config configs/train_resnet50.yaml` | Needs GPU run |
| Grad-CAM overlays (correct and incorrect) | `reports/explain/*.png` | `python -m training.explain --config configs/train_resnet50.yaml` | Needs GPU run |
| Explanation checks (stroke mass, deletion test) | `reports/explain_summary.csv`, `explain_deletion.html`, `explain_finding.md` | same command | Needs GPU run |
| Champion registered in MLflow | `mlruns/` (not committed) | `python -m training.register --config configs/export.yaml` | Needs GPU run |
| ONNX parity, size, latency, INT8 comparison | `backend/models/model_card.json` (`export` block) | `python -m training.export_onnx --config configs/export.yaml` | Dev model only |
| Tremor LightGBM cross-validated AUC | `backend/models/tremor_model_card.json` | `python -m tremor.train_lgbm --config configs/tremor_lgbm.yaml` | Real, on simulated data |
| SHAP summary and bar plots | `reports/tremor_shap_summary.png`, `reports/tremor_shap_bar.png` | same command | Real, on simulated data |
| Backend latency and peak memory | `PROGRESS.md` (Phase 9) | see "Measure the backend" below | Real (Windows, dev model) |

## Measure the backend

The latency and memory figures in `PROGRESS.md` came from a throwaway script (not committed) that
started the app in a venv built from only `backend/requirements.txt`, sent 8 requests each to
`/predict` (400 x 400 PNG, 8 x 8 occlusion grid) and `/tremor` (960 points), and read the process
peak working set with `psutil`. To get numbers for the deployment target, build the image
(`docker compose build backend`), run it with `docker run --memory 512m`, send the same requests and
watch `docker stats`. Record the image size with `docker images`.

## Honest notes for the report

- Report the dataset as 1264 images from 158 subjects and say folds are grouped by subject.
- Do not quote anything marked "Needs GPU run" until it is regenerated without `--fast`.
- The tremor LightGBM numbers are about the simulator. Say so wherever they appear.
- State domain shift (scans vs canvas) as the main limitation.
