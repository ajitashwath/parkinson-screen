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

## Phase 1: Data layer and preprocessing - not started

## Phase 2: Tremor analysis package - not started

## Phase 3: Training pipeline - not started

## Phase 4: Experiments - not started

## Phase 5: Explainability, export and registry - not started

## Phase 6: LightGBM tremor model and SHAP - not started

## Phase 7: Backend (FastAPI) - not started

## Phase 8: Frontend (Next.js) - not started

## Phase 9: Docker, CI and deployment - not started

## Phase 10: Documentation and report assets - not started
