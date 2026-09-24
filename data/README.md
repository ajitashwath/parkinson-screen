# data

This project does not ship any dataset. Download at least one of the
datasets below into `data/raw/`, matching the layout the code expects, then
run the audit script. `data/raw/` and `data/processed/` are gitignored;
only this file and `data/folds.csv` (written by the audit) are committed.

## Expected layout

```
data/raw/<dataset_name>/<spiral|wave>/<healthy|parkinson>/*.png
```

`<dataset_name>` is any folder name you choose per source (for example
`kaggle_parkinsons_drawings`, `handpd`, `pahaw`). Everything under it must
follow `<spiral|wave>/<healthy|parkinson>/`. If a downloaded dataset ships a
different layout (for example, separate `training/` and `testing/`
subfolders, or a single combined folder per class), reorganize the files
into this layout before running the audit; `training/data.py` does not
guess at other layouts, per this project's rule against inventing dataset
structure.

## Datasets

### Kaggle "Parkinson's Drawings"

A commonly used public dataset of spiral and wave tracings, split into
healthy and Parkinson's classes. Search Kaggle for "Parkinson's Drawings"
(spiral and wave, healthy vs Parkinson) to find it and confirm the current
URL and license yourself before downloading; Kaggle dataset slugs and
ownership can change, so this file intentionally does not hardcode one.
This dataset is not known to include subject IDs in its filenames; expect
`training/audit_data.py` to report `subject_ids_available: false` for it
and to fall back to hash-grouped folds. Accept the dataset's license on
Kaggle before downloading.

### HandPD / NewHandPD

A handwriting and drawing dataset from a Brazilian research group (UNESP),
collected specifically for Parkinson's screening research, including
spiral and meander (wave-like) exams. It is distributed by the research
group rather than an open marketplace and typically requires filling in a
request form and citing the associated paper. Search for "HandPD NewHandPD
Parkinson dataset" to find the current request page, and check whether it
includes subject-level identifiers in its file naming or metadata; if so,
set `subject_id_regex` in `configs/data.yaml` to match it once you know the
exact convention used by the files you receive.

### PaHaW (Parkinson's Disease Handwriting Database)

A handwriting database from Czech Technical University with tasks beyond
spirals and waves. Optional for this project; only use the spiral- and
wave-equivalent tasks if you include it, and only if you can map its task
labels to this project's `<spiral|wave>/<healthy|parkinson>` layout without
guessing. Access is by request from the original authors; search for
"PaHaW Parkinson's disease handwriting database" for the current contact
details.

## After downloading

```bash
python -m training.audit_data --config configs/data.yaml
```

This prints image counts, sizes, duplicate groups and whether subject IDs
were found, writes the same summary to `docs/DATASHEET.md`, and writes fold
assignments to `data/folds.csv`. Use `--fast` for a quick sample-based
check on a large dataset before running the full audit.

## Leakage note

If a dataset has no usable subject IDs, `training/data.py` groups folds by
perceptual image hash instead, which only protects against duplicate or
near-duplicate images splitting across train and test. It does not protect
against the same person's other, non-duplicate drawings appearing in both a
training and a test fold. `docs/DATASHEET.md` states loudly, per dataset,
which case applies. Prefer a dataset with real subject IDs when one is
available.
