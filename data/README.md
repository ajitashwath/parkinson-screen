# data

`data/raw/`, `data/raw/_downloads/` and `data/processed/` are gitignored;
only this file and `data/folds.csv` (written by the audit) are committed.
No drawing images are ever committed to this repository.

## Datasets actually integrated (HandPD and NewHandPD)

This project uses the real **HandPD** and **NewHandPD** datasets from the
Signal Processing group at São Paulo State University (UNESP), Botucatu
Medical School: 92 individuals (18 healthy, 74 Parkinson's) in HandPD and
66 individuals (35 healthy, 31 Parkinson's) in NewHandPD, each performing 4
spiral and 4 meander (wave-equivalent) drawing attempts on a printed
template. Dataset home page:
<https://wwwp.fc.unesp.br/~papa/pub/datasets/Handpd/>. The archives are
served directly over HTTP with no login, account or request form. No
formal license is posted; the page asks that you cite the associated
papers (CMPB-2016 for HandPD, SIBGRAPI-2016 for NewHandPD) if you use the
data, which this project does.

To reproduce from scratch:

```bash
mkdir -p data/raw/_downloads/NewHealthy data/raw/_downloads/NewPatients
cd data/raw/_downloads
curl -sSL -O https://wwwp.fc.unesp.br/~papa/pub/datasets/Handpd/Spiral_HandPD.zip
curl -sSL -O https://wwwp.fc.unesp.br/~papa/pub/datasets/Handpd/Meander_HandPD.zip
curl -sSL -o NewHealthy/HealthySpiral.zip https://wwwp.fc.unesp.br/~papa/pub/datasets/Handpd/NewHealthy/HealthySpiral.zip
curl -sSL -o NewHealthy/HealthyMeander.zip https://wwwp.fc.unesp.br/~papa/pub/datasets/Handpd/NewHealthy/HealthyMeander.zip
curl -sSL -o NewPatients/PatientSpiral.zip https://wwwp.fc.unesp.br/~papa/pub/datasets/Handpd/NewPatients/PatientSpiral.zip
curl -sSL -o NewPatients/PatientMeander.zip https://wwwp.fc.unesp.br/~papa/pub/datasets/Handpd/NewPatients/PatientMeander.zip
cd ../../..
python -m training.prepare_handpd --downloads-dir data/raw/_downloads --out-dir data/raw
python -m training.audit_data --config configs/data.yaml
```

The server is slow; the two big HandPD zips (132 MB and 90 MB) can take
several minutes each. Use `curl -C -` to resume a partial download if it
times out. `data/raw/_downloads/` (the raw zips, about 340 MB) can be
deleted once `prepare_handpd` has run successfully; the reorganized images
under `data/raw/handpd/` and `data/raw/newhandpd/` (about 340 MB) are what
training actually reads.

**Subject IDs come from filenames, not from the published CSVs.**
HandPD's own `Spiral_HandPD.csv` / `Meander_HandPD.csv` turned out to be
plain LibOPF feature exports with no image name or patient ID column at
all, despite the dataset page describing richer columns. NewHandPD's
`NewSpiral.csv` has a publishing bug: every patient row's `IMAGE_NAME`
points at a meander file regardless of which CSV it is in, and its
`ID_PATIENT` values overlap between the healthy and patient classes, which
a real per-class patient code should not do. Both were caught by joining
the CSVs against the real archive contents before trusting them (see
`docs/ASSUMPTIONS.md`). `training/prepare_handpd.py` instead derives the
subject ID directly from each dataset's own filename convention, which is
consistent with no exceptions in the archives: `0068-1.jpg` to subject
`0068` for HandPD, `sp1-H9.jpg` / `mea1-H9.jpg` to subject `H9` for
NewHandPD. It writes this as `data/raw/<dataset>/metadata.csv` (columns
`kind`, `image_name`, `subject_id`), which `training/data.py` reads in
preference to guessing a filename regex. The result:
`docs/DATASHEET.md` reports `subject_ids_available: true` and folds are
grouped by real subject, not by the weaker duplicate-hash fallback.

## Expected layout

```
data/raw/<dataset_name>/<spiral|wave>/<healthy|parkinson>/*.jpg
```

`<dataset_name>` is any folder name (`handpd` and `newhandpd` for the
datasets above). Everything under it must follow
`<spiral|wave>/<healthy|parkinson>/`. An optional
`data/raw/<dataset_name>/metadata.csv` (columns `kind`, `image_name`,
`subject_id`) gives exact subject IDs sourced from the dataset itself; see
`training/prepare_handpd.py` for how HandPD's is built. Without one,
`training/data.py` falls back to a caller-supplied `subject_id_regex` in
`configs/data.yaml`, or to `None` if that does not match either.

## Other datasets you can add

### Kaggle "Parkinson's Drawings"

A commonly used public dataset of spiral and wave tracings, split into
healthy and Parkinson's classes. It requires a Kaggle account to download
(via the website or `kaggle datasets download`), which this project does
not do on your behalf. Search Kaggle for "Parkinson's Drawings" to find it
and confirm the current URL and license yourself; Kaggle dataset slugs and
ownership can change, so this file intentionally does not hardcode one.
This dataset is not known to include subject IDs; expect
`training/audit_data.py` to fall back to hash-grouped folds for it, unless
you write a `metadata.csv` for it yourself.

### PaHaW (Parkinson's Disease Handwriting Database)

A handwriting database from Czech Technical University with tasks beyond
spirals and waves. Optional for this project; only use the spiral- and
wave-equivalent tasks if you include it, and only if you can map its task
labels to this project's `<spiral|wave>/<healthy|parkinson>` layout without
guessing. Access is by request from the original authors; search for
"PaHaW Parkinson's disease handwriting database" for the current contact
details.

## After adding a dataset

```bash
python -m training.audit_data --config configs/data.yaml
```

This prints image counts, sizes, duplicate groups and whether subject IDs
were found, writes the same summary to `docs/DATASHEET.md`, and writes fold
assignments to `data/folds.csv`. Use `--fast` for a quick sample-based
check on a large dataset before running the full audit.

## Leakage note

If a dataset has no usable subject IDs (no `metadata.csv` and no matching
`subject_id_regex`), `training/data.py` groups folds by perceptual image
hash instead, which only protects against duplicate or near-duplicate
images splitting across train and test. It does not protect against the
same person's other, non-duplicate drawings appearing in both a training
and a test fold. `docs/DATASHEET.md` states loudly, per dataset, which case
applies.
