"""Request and response models. Every response carries the disclaimer."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.settings import DISCLAIMER

ScreeningLabel = Literal["lower_concern", "higher_concern", "uncertain"]
DrawingKind = Literal["spiral", "wave"]
PointerKind = Literal["mouse", "pen", "touch"]

MAX_COORDINATE = 1_000_000.0
MAX_TIME_MS = 3_600_000.0
MAX_BASE64_CHARS = 3_000_000
DEVICE_ID_PATTERN = r"^[A-Za-z0-9_-]{8,64}$"


class Disclaimed(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    disclaimer: str = DISCLAIMER


class ErrorResponse(Disclaimed):
    code: str
    detail: str


class HealthResponse(Disclaimed):
    status: Literal["ok", "unavailable"]
    imageModel: str
    tremorModel: str
    message: str
    imageModelLoaded: bool
    tremorModelLoaded: bool


class PredictRequest(BaseModel):
    image_base64: str = Field(min_length=16, max_length=MAX_BASE64_CHARS)
    kind: DrawingKind


class OcclusionMap(BaseModel):
    method: Literal["occlusion"] = "occlusion"
    description: str
    grid_size: int
    values: list[list[float]]
    raw_probability_drop: list[list[float]]


class PredictResponse(Disclaimed):
    probability: float
    label: ScreeningLabel
    threshold: float
    uncertain_margin: float
    model_version: str
    development_model: bool
    provenance: str
    kind: DrawingKind
    occlusion_map: OcclusionMap
    warnings: list[str]


class PointIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    x: float = Field(ge=-MAX_COORDINATE, le=MAX_COORDINATE)
    y: float = Field(ge=-MAX_COORDINATE, le=MAX_COORDINATE)
    t_ms: float = Field(ge=0.0, le=MAX_TIME_MS)
    pressure: float | None = Field(default=None, ge=0.0, le=1.0)
    pointer_type: str = Field(default="mouse", max_length=16)
    pen_lift: bool = False

    @field_validator("x", "y", "t_ms")
    @classmethod
    def must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("must be a finite number")
        return value


class TremorRequest(BaseModel):
    points: list[PointIn]
    pointer_type: str = "mouse"


class Contribution(BaseModel):
    feature: str
    value: float
    explanation: str


class TremorResponse(Disclaimed):
    features: dict[str, float]
    warnings: list[str]
    score: float | None
    model_loaded: bool
    top_contributions: list[Contribution]
    provenance: str
    trained_on: str


class SessionCreate(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    device_id: str = Field(pattern=DEVICE_ID_PATTERN)
    kind: DrawingKind
    label: ScreeningLabel
    probability: float = Field(ge=0.0, le=1.0)
    tremor_score: float | None = Field(default=None, ge=0.0, le=1.0)
    spiral_deviation: float | None = Field(default=None, ge=0.0, le=1000.0)
    pointer_type: str = Field(default="mouse", max_length=16)
    model_version: str = Field(default="", max_length=64)
    features: dict[str, float] = Field(default_factory=dict, max_length=64)

    @field_validator("features")
    @classmethod
    def features_must_be_finite(cls, value: dict[str, float]) -> dict[str, float]:
        if not all(math.isfinite(v) for v in value.values()):
            raise ValueError("feature values must be finite")
        return value


class SessionRecord(Disclaimed):
    id: str
    createdAt: str
    kind: DrawingKind
    label: ScreeningLabel
    probability: float
    tremorScore: float | None
    spiralDeviation: float | None
    pointerType: str


class SessionList(Disclaimed):
    sessions: list[SessionRecord]


class DeleteResponse(Disclaimed):
    deleted: int
