# Decisions

Engineering decisions made along the way, with the reasoning, so later phases
do not silently contradict earlier ones.

## Phase 0

- **Two dependency files, never merged.** `requirements-train.txt` for
  training and dev tooling, `backend/requirements.txt` for the FastAPI
  service. No torch and no `shap` package in the backend file, per
  `CLAUDE.md` section 2, so the backend Docker image stays small enough for
  Render's free tier.
- **Tool config lives in one `pyproject.toml`** (`[tool.ruff]`, `[tool.mypy]`,
  `[tool.pytest.ini_options]`) rather than separate `ruff.toml` / `mypy.ini`
  files, to keep Phase 0 config in one place.
- **Lint and type-check scope excludes `data/`, `frontend/`, `reports/`,
  `mlruns/` and `wandb/`.** Those hold generated artifacts, JS/TS code (its
  own ESLint/tsc will cover it from Phase 8), or third-party output, not
  first-party Python.
