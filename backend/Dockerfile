# Build from the repository root: docker build -f backend/Dockerfile .
# CPU only. No PyTorch, no shap: models are served with onnxruntime and LightGBM.

FROM python:3.11-slim AS builder
WORKDIR /build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# Serving needs only the runtime packages. httpx and pytest are test tools.
COPY backend/requirements.txt .
RUN grep -vE '^(httpx|pytest)' requirements.txt > runtime.txt && pip install -r runtime.txt

FROM python:3.11-slim AS runtime
# LightGBM needs the OpenMP runtime, which the slim image does not ship.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY backend/*.py backend/
# The one shared preprocessing module and the tremor feature code, copied in
# so serving preprocesses exactly like training without the training package.
COPY common/__init__.py common/preprocess.py common/
COPY tremor/__init__.py tremor/features.py tremor/
COPY backend/models/model.onnx backend/models/model_card.json backend/models/threshold.json \
     backend/models/tremor_lgbm.txt backend/models/tremor_model_card.json backend/models/

USER app
ENV DATABASE_URL=sqlite:////tmp/screening.db PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os,urllib.request as u; u.urlopen('http://127.0.0.1:%s/api/v1/health' % os.environ['PORT'], timeout=4)"
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}"]
