# Code Review: Parkinson Screen

## Summary

**What it is.** A research and demo system with three parts. A CNN reads a scanned or canvas-drawn spiral or wave. A signal-processing pipeline measures tremor from *how* the drawing was made (timestamps and positions). A web app shows both. The backend is a lean FastAPI service that serves an ONNX image model and a LightGBM tremor model on CPU, with no PyTorch in the serving image.

**Verdict on the code.** Well above typical student or demo quality. The architecture is deliberate, the tests are real, the documentation is unusually honest, and the data-leakage control is correct (I verified it, see section 3). The weaknesses are concentrated in the backend's *robustness edges* (a memory-exhaustion path, a tremor score that fires on garbage input, size and rate limits that are easy to get around) and in a few *statistical shortcuts* in the evaluation code. None of them is hard to fix, and several can be fixed before you present.

**Most important caveat for your talk.** The image model shipped in the repo is a development model (one epoch, random init, tiny sample) and the tremor model is trained on simulated data only. The project says this itself everywhere. No accuracy number in the repo is real yet. Build the presentation (see PRESENTATION.md) around the *engineering*, not around model performance.

---

## 1. Scorecard

| Area | Rating | One-line reason |
|---|---|---|
| Architecture and structure | Strong | Clean separation of common / training / tremor / backend; serving image deliberately minimal |
| Code style and readability | Strong | Consistent, typed, small functions, docstrings that explain *why*; ruff clean |
| Backend correctness (happy path) | Strong | 76 tests pass; behaviour matches the spec in the docs |
| Backend correctness (edge cases) | Needs work | Four real defects found by probing (section 4: H1 to H3 and the M-series) |
| Security and hardening | Adequate for a demo | Good basics; gaps in size, memory and rate limiting |
| Testing | Strong | Parity, determinism, train/serve identity, contribution-sum tests are all the right ones |
| ML methodology | Good design, unfinished | Leakage control correct; some statistical shortcuts in evaluation code |
| DevOps | Good, unproven | Multi-stage Docker, compose, CI, Render blueprint, but never run |
| Frontend (brief) | Good | Typed API client with a clearly-labelled mock adapter; thin test coverage |
| Documentation honesty | Excellent | Limits stated in README, model cards, handoff; `TODO(run-on-gpu)` instead of fake numbers |

## 2. How I reviewed this

**Executed**
- 76 tests: `test_backend.py`, `test_preprocess.py`, `test_tremor_features.py`. All pass.
- `ruff check .` on the whole repository: clean.
- `mypy backend common`: clean.
- Targeted probes against the running app (large images, degenerate strokes, chunked bodies, proxy headers, timestamps, logging format).
- Integrity checks on `data/folds.csv`.
- Tremor frequency recovery and shipped-model behaviour checks.

**Read in full:** everything in `backend/`, `common/`, `tremor/features.py`, `training/data.py`, `training/metrics.py`, deployment files, CI, docs.
**Skimmed:** `training/train.py` (first 200 lines), the shortcut-check decision logic, the structure of `export_onnx.py`, `explain.py`, `register.py`, frontend API and storage layers.

**Not run:** the PyTorch-dependent tests (about 185 functions, need `torch` and `plotly`; collection failed for `test_tremor_validate.py` on a missing plotly), frontend tests and build, Docker build, CI. I used Python 3.12 where the project targets 3.11. Findings in the training code are therefore from reading, not execution.

## 3. What is done well

1. **Right abstractions.** `create_app(settings)` factory, `lifespan` for model loading, dependency-injected DB session, `LoadedModels` container that tolerates missing models. This is what makes the backend testable.
2. **Single preprocessing path** with a bit-exact train/serve test. This is the best single design decision in the repository.
3. **Honest model labelling.** `development_model` flows from the model card to the API response, the warning list and the UI. `trained_on: synthetic` is returned with every tremor score.
4. **Exact explanations without extra dependencies.** `pred_contrib` plus a test that contributions sum to the raw score.
5. **Memory awareness.** Chunked inference, capped passes, disabled ONNX arena, with the reason documented next to the code.
6. **Validation at the boundary.** Finite-number validators, range limits, regex-checked device ID, strict base64, one error format.
7. **Leakage control that actually holds.** Verified: 0 subjects in more than one fold, 0 subjects with mixed labels, no ID collisions across the two datasets, 31 or 32 subjects per fold.
8. **Test quality.** Tests assert properties (determinism, parity, isolation between devices, cap per device), not just status codes.
9. **Thoughtful signal processing.** The vx/vy spectral decision is explained, tested, and verified here (0.28 Hz max recovery error).
10. **Reproducibility scaffolding.** Seeds, config files, `--fast` mode, `REPORT_ASSETS.md` mapping every figure to its command.

## 4. Findings, backend (by severity)

Each item has evidence and a fix. "Verified" means I reproduced it; "From reading" means I did not.

### High

**H1. A tiny image can exhaust memory (Verified)**
- Only the *byte* size is capped (2 MB decoded). Pillow decodes a small, highly compressible PNG into a huge pixel buffer.
- A 10,000 x 10,000 grayscale PNG is **117 KB** on the wire. Posting it to `/predict` returned 200, and process peak memory went from **279 MB to 744 MB**. Render's free tier kills a process at 512 MB.
- Pillow only *warns* above 89 M pixels and errors above 178 M, so this range slips through. (A 20,000 x 20,000 image was rejected correctly.)
- This one request defeats the memory-saving design the whole backend is built around.
- **Fix:** check the header-declared size before decoding (`Image.open` is lazy):

```python
MAX_PIXELS = 16_000_000

def to_model_input(image_bytes: bytes) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(image_bytes)) as probe:
            if probe.width * probe.height > MAX_PIXELS:
                raise ImageDecodeError("Image dimensions are too large.")
        image = preprocess_image(image_bytes, PreprocessConfig(size=IMAGE_SIZE))
    except ImageDecodeError:
        raise
    except Exception as error:
        raise ImageDecodeError("The image could not be read as a PNG or JPEG.") from error
    return np.transpose(image, (2, 0, 1))[None].astype(np.float32)
```
Add a test that posts a large-dimension, small-byte PNG and expects 422.

**H2. The tremor score fires on input that is not a tremor trace (Verified)**
- Validation produces warnings, but the score is computed regardless.
- Results from the shipped booster:
  - 60 points that all share one timestamp: `score = 0.9999` (and a divide-by-zero warning in `compute_kinematics`, because `dt = 0`)
  - a perfectly straight horizontal line: `score = 0.99998`
  - a noise-free spiral with *no* tremor: `score = 1.0`; the same spiral with 0.3 px of noise: `score = 0.0`
- Cause: the model is trained on simulator output only, so out-of-distribution input gets extreme, confident scores. The model card admits the label is "tremor was injected", but the API still presents a number.
- Impact: the UI can show a very high tremor score for a trace that is not a spiral at all.
- **Fix:** gate the score. Return `score = None` when the quality warnings fire or when the core spectral features are missing.

```python
REQUIRED = ("tremor_band_ratio", "peak_freq_hz", "spectral_entropy", "mean_speed")

def analyse_tremor(points, model):
    result = compute_features(points)
    features = finite_features(result)
    usable = all(name in features for name in REQUIRED) and not any(
        w.startswith(("too few points", "duration too short", "fewer than 2"))
        for w in result.warnings
    )
    ...
    if model is not None and usable:
        score, contributions = model.score_and_contributions(result.features)
```
Also guard `dt <= 0` in `compute_kinematics`, and consider a simple range check against the training distribution (min and max per feature) to flag out-of-distribution input.

**H3. Rate limiting is unlikely to work as intended behind Render's proxy, and it leaks memory (Rate-key issue: from reading. Memory growth: Verified)**
- The limiter keys on `request.client.host`. Behind a reverse proxy that is the proxy's address unless uvicorn is told to trust forwarded headers. The Dockerfile does not set `--proxy-headers` / `--forwarded-allow-ips`, and I confirmed `X-Forwarded-For` is ignored.
- Likely result on Render: **every visitor shares one 60-requests-per-minute bucket**. Each submission costs three calls (predict, tremor, sessions), so about 20 submissions per minute for the whole audience. A classroom demo could trip this.
- Separately, `RateLimiter.hits` never evicts a key unless that same client returns. 10,000 distinct clients left 10,000 entries in memory.
- It is also per-process and endpoint-blind: a heavy `/predict` (about 2.5 s of CPU) costs the same budget as a cheap `/health`-like call.
- **Fix:** run uvicorn with `--proxy-headers --forwarded-allow-ips="*"` (only if you trust the platform's proxy), evict stale keys (periodic sweep or a TTL cache), and give `/predict` a lower limit than other routes.

### Medium

**M1. The body-size limit only checks the `Content-Length` header (Verified)**
- A chunked upload (no `Content-Length`) with a body larger than the limit returned **200** in my test. The server reads the whole body into memory before Pydantic runs.
- `BaseHTTPMiddleware` also buffers responses and is known to interact badly with streaming and background tasks.
- **Fix:** a small pure-ASGI middleware that counts bytes in `receive()` and aborts with 413, or enforce the limit at the proxy. Also constrain the `points` list length at the schema level so a 20,000+ point body is rejected during parsing, not after.

**M2. The session endpoint trusts client-supplied results (Verified)**
- `POST /sessions` stores whatever `label`, `probability`, `tremor_score` and `features` the client sends. A contradictory pair (`label=higher_concern`, `probability=0.01`) was accepted with 201.
- The stored history is therefore not evidence of what the model produced, and anyone can write arbitrary rows for any device ID.
- Acceptable for a convenience history; not acceptable if the data is ever analysed. **Fix:** have `/predict` return a short-lived signed result token (HMAC over the response fields) that `/sessions` verifies, or store the result server-side at predict time and return a session ID.

**M3. Logging is not valid JSON (Verified)**
- The format string builds JSON by hand with `%(message)s`. Any message containing a double quote breaks the line. This already happens: the access-style lines from `httpx` in my run were `"message": "HTTP Request: POST ... "HTTP/1.1 201 Created""`.
- `logging.basicConfig` inside `create_app` is also silently a no-op if logging is already configured (uvicorn configures it first).
- **Fix:** a formatter that calls `json.dumps` on a dict, or `python-json-logger` / `structlog`. Add request ID and latency logging.

**M4. `/health` returns 200 even when models failed to load (Verified by test)**
- Intentional (so the UI can show a message), but Render's `healthCheckPath` and the Docker `HEALTHCHECK` both pass while `/predict` returns 503, so the platform will route traffic to a broken instance.
- **Fix:** keep `/health` as is for the UI, and add `/ready` that returns 503 when either model is missing; point the platform health checks at `/ready`.

**M5. Timestamps are inconsistent between create and list (Verified on SQLite)**
- `POST /sessions` returns `createdAt` with `+00:00`; `GET /sessions` on SQLite returns it **without** a timezone, because SQLite drops tzinfo. Postgres returns aware values.
- The frontend sorts history with `localeCompare` on these strings, so mixed formats can order incorrectly.
- **Fix:** always serialise UTC with a `Z` suffix (normalise in `session_record`), or use a `TypeDecorator` that guarantees aware UTC.

**M6. The explanation is the capacity bottleneck**
- About 2.5 s per `/predict` on a 1 vCPU sandbox (65 passes), and Render's free tier is smaller. Each request occupies a thread-pool worker and a CPU for the whole time, and the default limit (60 per minute per client) allows roughly 150 CPU-seconds per minute from one client.
- `intra_op_num_threads = 2` is hard-coded.
- **Fix:** make the map optional (`?explain=false` or a separate endpoint), cache by image hash, expose thread count in settings, and weight the rate limit by endpoint.

**M7. No schema migrations**
- `Base.metadata.create_all` runs at startup. Any future column change will break an existing database. There is also a redundant index: `device_id` has `index=True` *and* a composite index that already covers it.
- **Fix:** add Alembic when the schema first changes; drop the duplicate index.

### Low

| ID | Finding | Fix |
|---|---|---|
| L1 | `TremorRequest.pointer_type` is accepted but never used (per-point `pointer_type` is what counts); an unknown point type is silently coerced to `"mouse"` | Remove the field or use it; reject unknown values instead of coercing |
| L2 | Three `# type: ignore[arg-type]` in `main.py` come from storing `kind`/`label` as plain `str` columns | Validate on write and use `typing.cast` or Literal-typed accessors |
| L3 | `booster.predict` is called twice per request (score, then contributions) | The score equals the sigmoid of the summed contributions; one call is enough |
| L4 | `resample_uniform` returns a pressure grid that nothing uses | Drop it, or use it for pen-only pressure features |
| L5 | Strokes with pen lifts are linearly interpolated across the gap during resampling, adding a fake fast segment to the spectrum | Segment the stroke at pen lifts before resampling |
| L6 | `app = create_app()` runs at import time | Fine for uvicorn; consider `uvicorn --factory` if import side effects ever matter |
| L7 | `GET /sessions` is unpaginated (bounded by the 500 cap) | Add `limit`/`offset` if the cap is raised |
| L8 | Dockerfile `CMD ["sh", "-c", "uvicorn ..."]` without `exec`; no `--proxy-headers`; test packages removed with a `grep -vE` on the requirements file | Use `exec uvicorn ...`; split into `requirements.txt` and `requirements-dev.txt` |
| L9 | `docker-compose.yml` starts the frontend after the backend *starts*, not when it is *healthy* | Add a backend healthcheck and `condition: service_healthy` |
| L10 | CI builds the Docker images but never starts them | Add a smoke step: run the container, `curl /api/v1/health` |
| L11 | `.github/workflows/ci.yml` installs the training stack via a `grep -v` workaround for `grad-cam` | Pin in a separate extras file so the install is declarative |
| L12 | A blank line is missing before `app.add_middleware(...)` in `main.py` (style only) | Cosmetic |

## 5. Findings, ML and evaluation code (from reading)

**ML1. Confidence intervals ignore the subject structure.**
`bootstrap_auc_ci` and `paired_bootstrap_comparison` resample *images* independently, but each subject contributes 8 correlated drawings. That makes intervals too narrow and p-values too small. Resample **subjects** (cluster bootstrap). Also report subject-level predictions (average a person's drawings), since that is the unit a screening decision would apply to.

**ML2. The shortcut check has a one-sided decision rule.**
`leak_detected = background["mean_auc_ci_lower"] > LEAK_AUC_LOWER_BOUND` only tests for AUC *above* chance. The committed report shows the background-only model at AUC **0.100**, and the rule prints "no shortcut detected". An AUC far *below* 0.5 is just as informative (the background predicts the label in the opposite direction), and with 2 folds and 5 test images it is mostly noise. Two corrections are needed: test `|AUC - 0.5|`, and average the *predictions* or use a proper interval, because the average of per-fold lower bounds is not a confidence bound on the average AUC.

**ML3. Auto-generated "findings" conclude more than the data allow.**
`shortcut_check_finding.md` and `explain_finding.md` state conclusions ("no shortcut detected", "the model attends to strokes") from `--fast` runs of a randomly initialised model. Both files carry a development-run note, which is good, but the sentence a reader sees is still a conclusion. Make the generator refuse to conclude below a minimum number of folds and test samples.

**ML4. The tremor model is brittle (see H2).**
Class separation is by construction, so the 0.998 cross-validated AUC says nothing about real tremor. The model card says this, correctly. The shipped booster also mis-scores traces outside the simulator's noise range, so do not present its score as a measurement.

**ML5. Decision threshold and "probability".**
- The threshold is chosen on a small validation split (high variance with few subjects).
- No calibration step is applied (Brier score is reported, but no Platt or isotonic fit), yet the API returns a field called `probability` and the UI treats it as one.
- The ±0.05 "uncertain" band is a fixed constant, not derived from calibration or from the threshold's uncertainty.
- The README's wording is careful, but the field name overstates.

**ML6. Normalisation.**
Pretrained backbones receive inputs in [0, 1] with no ImageNet mean/std, by documented choice. It is consistent between training and serving, so it is not a bug, but it is a departure from what pretrained weights expect, and it is worth one sentence in the report if an ablation exists.

**ML7. Dataset facts to state accurately.**
The perceptual-hash "duplicate" count (246 groups, 536 images) is expected because subjects retrace the same printed template; the datasheet explains this. Subject grouping, not the hash, is what protects against leakage.

## 6. Frontend (brief)

**Good:** a typed API client with explicit mapping functions, a clearly-labelled demo adapter, graceful degradation (tremor failure becomes a warning, history failure never hides the result), device ID in `localStorage`, tests for the API client, storage, formatting and upload.

**Issues**
- `predict`, `tremor` and `sessions` are called one after another. `predict` and `tremor` are independent and could run in parallel with `Promise.all`, cutting perceived latency.
- The client-side result ID (`result-<timestamp>`) differs from the server's session ID, which is discarded. History from `localStorage` and from the API cannot be reliably de-duplicated.
- If `NEXT_PUBLIC_API_URL` is missing at build time the app silently falls back to mock mode. It is labelled, but a misconfigured production build would show fake results; fail the build in production instead.
- Only about 10 frontend tests; the drawing canvas, result view and history page are untested.

## 7. Repository hygiene

- `data/folds.csv` contains Windows-style paths (`data\raw\handpd\...`). `HANDOFF.md` says training will run on Colab or Kaggle (Linux), where these paths will not resolve. Regenerate with `audit_data` on the target machine or store relative POSIX paths.
- `backend/models/model_card.json` has CRLF line endings and a Windows path (`backend\\models\\model_int8.onnx`) for a file that is not committed.
- `.gitattributes` exists; check it normalises line endings.
- `CLAUDE.md`, `PROGRESS.md` (58 KB) and `HANDOFF.md` are in the repo root. Decide deliberately whether they belong in a submission.
- No `LICENSE` file; the datasets have their own citation requirements (listed in `data/README.md`).

## 8. Prioritised fix list

**Before you present (about 1 to 2 hours, backend only)**
1. Add the pixel-count cap (H1) and a test.
2. Gate the tremor score on input quality and guard `dt <= 0` (H2) and a test using identical timestamps.
3. Replace the hand-built JSON log format (M3).
4. Add `--proxy-headers` to the Dockerfile `CMD` and evict old rate-limiter keys (H3).
5. Run `docker build` and `docker run --memory 512m`, then record the real image size and peak memory. This is an open item in `HANDOFF.md`, and it gives you a measured number to quote instead of an estimate.

**Before you claim any result**
6. Run the full GPU experiments listed in `HANDOFF.md`.
7. Switch to subject-level bootstrap and subject-level metrics (ML1).
8. Fix the two-sided shortcut test (ML2) and regenerate the findings with a minimum-sample rule (ML3).

**For a real deployment**
9. Pure-ASGI body-size limit (M1), signed results for `/sessions` (M2), `/ready` endpoint (M4), UTC-normalised timestamps (M5), optional or cached explanation (M6), Alembic (M7).
10. Authentication or at least a server-issued, rotating device token instead of a client-chosen ID in the query string.

---

*Review basis: I executed 76 tests, ruff on the whole repo, mypy on `backend` and `common`, and targeted probes. Training code, frontend tests, Docker and CI were reviewed by reading only.*
