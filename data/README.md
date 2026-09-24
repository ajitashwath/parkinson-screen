# data

Dataset download instructions, the audit summary and fold assignments are written here in
Phase 1 (see `CLAUDE.md`, Phase 1). This stub exists so the directory is tracked by git before
that phase runs.

Expected layout once datasets are downloaded:

```
data/raw/<dataset>/<spiral|wave>/<healthy|parkinson>/*.png
```

`data/raw/` and `data/processed/` are gitignored. Only `data/README.md` and `data/folds.csv`
(added in Phase 1) are committed.
