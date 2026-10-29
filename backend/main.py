"""FastAPI app. Run from the repository root: uvicorn backend.main:app."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from sqlalchemy.orm import Session
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend import database
from backend.inference import (
    ImageDecodeError,
    LoadedModels,
    analyse_tremor,
    decode_image_base64,
    load_models,
    normalize_drop,
    to_model_input,
)
from backend.schemas import (
    DEVICE_ID_PATTERN,
    Contribution,
    DeleteResponse,
    HealthResponse,
    OcclusionMap,
    PredictRequest,
    PredictResponse,
    ScreeningLabel,
    SessionCreate,
    SessionList,
    SessionRecord,
    TremorRequest,
    TremorResponse,
)
from backend.security import LimitsMiddleware, RateLimiter, error_response
from backend.settings import DISCLAIMER, Settings, get_settings
from tremor.features import StrokePoint

logger = logging.getLogger("backend")

OCCLUSION_DESCRIPTION = (
    "Occlusion map: how much the probability drops when each patch of the drawing is "
    "covered. It shows where the model looked, not why it is right."
)
DEV_MODEL_WARNING = (
    "Development model: trained on a small sample to prove the pipeline runs. "
    "Not a validated screening model."
)
POINTER_KINDS = {"mouse", "pen", "touch"}


def screening_label(probability: float, threshold: float, margin: float) -> ScreeningLabel:
    if abs(probability - threshold) <= margin:
        return "uncertain"
    return "higher_concern" if probability >= threshold else "lower_concern"


def to_stroke_points(request: TremorRequest) -> list[StrokePoint]:
    return [
        StrokePoint(
            x=p.x,
            y=p.y,
            t_ms=p.t_ms,
            pressure=p.pressure,
            pointer_type=p.pointer_type if p.pointer_type in POINTER_KINDS else "mouse",  # type: ignore[arg-type]
            pen_lift=p.pen_lift,
        )
        for p in request.points
    ]


def settings_of(request: Request) -> Settings:
    return request.app.state.settings


def models_of(request: Request) -> LoadedModels:
    return request.app.state.models


def db_session(request: Request) -> Iterator[Session]:
    with request.app.state.session_factory() as session:
        yield session


Db = Annotated[Session, Depends(db_session)]
DeviceId = Annotated[str, Query(pattern=DEVICE_ID_PATTERN)]


def build_router() -> APIRouter:
    router = APIRouter(prefix="/api/v1")

    @router.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        models = models_of(request)
        image, tremor = models.image, models.tremor
        ready = image is not None and tremor is not None
        if ready:
            message = "Models loaded."
            if image is not None and image.development:
                message = "Models loaded. The image model is a development model."
        else:
            message = " ".join(models.messages) or "A model is not loaded."
        return HealthResponse(
            status="ok" if ready else "unavailable",
            imageModel=image.version if image else "not loaded",
            tremorModel=str(tremor.card.get("model_name", "tremor")) if tremor else "not loaded",
            message=message,
            imageModelLoaded=image is not None,
            tremorModelLoaded=tremor is not None,
        )

    @router.post("/predict", response_model=PredictResponse)
    def predict(body: PredictRequest, request: Request) -> PredictResponse:
        settings = settings_of(request)
        image_model = models_of(request).image
        if image_model is None:
            raise HTTPException(503, "The image model is not loaded.")
        try:
            raw = decode_image_base64(body.image_base64, settings.max_image_bytes)
            model_input = to_model_input(raw)
        except ImageDecodeError as error:
            raise HTTPException(422, str(error)) from error

        result = image_model.predict_with_occlusion(
            model_input, settings.occlusion_grid_size, settings.max_occlusion_passes
        )
        warnings = [DEV_MODEL_WARNING] if image_model.development else []
        return PredictResponse(
            probability=result.probability,
            label=screening_label(
                result.probability, image_model.threshold, settings.uncertain_margin
            ),
            threshold=image_model.threshold,
            uncertain_margin=settings.uncertain_margin,
            model_version=image_model.version,
            development_model=image_model.development,
            provenance=(
                "Image model trained on HandPD and NewHandPD scanned drawings; "
                "canvas drawings differ from scans."
            ),
            kind=body.kind,
            occlusion_map=OcclusionMap(
                description=OCCLUSION_DESCRIPTION,
                grid_size=result.grid_size,
                values=normalize_drop(result.probability_drop).tolist(),
                raw_probability_drop=result.probability_drop.tolist(),
            ),
            warnings=warnings,
        )

    @router.post("/tremor", response_model=TremorResponse)
    def tremor(body: TremorRequest, request: Request) -> TremorResponse:
        settings = settings_of(request)
        if len(body.points) > settings.max_points:
            raise HTTPException(422, f"At most {settings.max_points} points are accepted.")
        model = models_of(request).tremor
        output = analyse_tremor(to_stroke_points(body), model)
        warnings = list(output.warnings)
        if model is None:
            warnings.append("The tremor model is not loaded; showing features only.")
        return TremorResponse(
            features=output.features,
            warnings=warnings,
            score=output.score,
            model_loaded=model is not None,
            top_contributions=[Contribution(**c) for c in output.contributions],
            provenance=model.provenance if model else "No tremor model loaded",
            trained_on=model.trained_on if model else "none",
        )

    @router.post("/sessions", response_model=SessionRecord, status_code=201)
    def create_session(body: SessionCreate, request: Request, db: Db) -> SessionRecord:
        row = database.add_session(db, body, settings_of(request).max_sessions_per_device)
        return session_record(row)

    @router.get("/sessions", response_model=SessionList)
    def get_sessions(device_id: DeviceId, db: Db) -> SessionList:
        rows = database.list_sessions(db, device_id)
        return SessionList(sessions=[session_record(r) for r in rows])

    @router.delete("/sessions", response_model=DeleteResponse)
    def remove_sessions(device_id: DeviceId, db: Db) -> DeleteResponse:
        return DeleteResponse(deleted=database.delete_sessions(db, device_id))

    return router


def session_record(row: database.SessionRow) -> SessionRecord:
    return SessionRecord(
        id=row.id,
        createdAt=row.created_at.isoformat(),
        kind=row.kind,  # type: ignore[arg-type]
        label=row.label,  # type: ignore[arg-type]
        probability=row.probability,
        tremorScore=row.tremor_score,
        spiralDeviation=row.spiral_deviation,
        pointerType=row.pointer_type,
    )


def validation_detail(error: RequestValidationError) -> str:
    parts = []
    for item in error.errors()[:5]:
        location = ".".join(str(p) for p in item["loc"] if p != "body")
        parts.append(f"{location}: {item['msg']}" if location else str(item["msg"]))
    return "; ".join(parts) or "Invalid request."


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format='{"time": "%(asctime)s", "level": "%(levelname)s", '
        '"name": "%(name)s", "message": "%(message)s"}',
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.models = load_models(settings.model_dir, settings.image_model_file)
        engine = database.make_engine(settings.database_url)
        app.state.session_factory = database.make_session_factory(engine)
        logger.info("startup complete: %s", app.state.models.messages or "models loaded")
        yield
        engine.dispose()

    app = FastAPI(title="Parkinson's screening aid API", version="1.0.0", lifespan=lifespan)
    app.state.settings = settings

    @app.get("/")
    def root() -> dict[str, str]:
        return {
            "name": "Parkinson's screening aid API",
            "status": "ok",
            "docs": "/docs",
            "health": "/api/v1/health",
            "disclaimer": DISCLAIMER,
        }

    @app.get("/health")
    def health_alias() -> RedirectResponse:
        return RedirectResponse(url="/api/v1/health")
    app.add_middleware(
        LimitsMiddleware,
        max_request_bytes=settings.max_request_bytes,
        limiter=RateLimiter(settings.rate_limit_per_minute),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.origins(),
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type"],
    )

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(request: Request, error: RequestValidationError) -> JSONResponse:
        return error_response(422, "invalid_request", validation_detail(error))

    @app.exception_handler(StarletteHTTPException)
    async def on_http_error(request: Request, error: StarletteHTTPException) -> JSONResponse:
        return error_response(error.status_code, f"http_{error.status_code}", str(error.detail))

    @app.exception_handler(Exception)
    async def on_unexpected_error(request: Request, error: Exception) -> JSONResponse:
        logger.exception("unhandled error on %s", request.url.path)
        return error_response(500, "internal_error", "Something went wrong on the server.")

    app.include_router(build_router())
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
