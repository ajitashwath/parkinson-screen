"""Backend configuration, read from environment variables (see .env.example)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DISCLAIMER = (
    "Screening aid, not a diagnosis. Possible signs only; not clinically validated. "
    "Talk to a clinician about any health concern."
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", protected_namespaces=("settings_",)
    )

    database_url: str = "sqlite:///./screening.db"
    allowed_origins: str = "http://localhost:3000"
    model_dir: Path = Path(__file__).parent / "models"
    image_model_file: str = "model.onnx"
    uncertain_margin: float = Field(default=0.05, ge=0.0, le=0.5)
    occlusion_grid_size: int = Field(default=8, ge=1, le=32)
    max_occlusion_passes: int = Field(default=65, ge=1, le=1024)
    max_image_bytes: int = 2 * 1024 * 1024
    max_points: int = 20_000
    max_request_bytes: int = 4 * 1024 * 1024
    rate_limit_per_minute: int = Field(default=60, ge=0)
    max_sessions_per_device: int = 500
    log_level: str = "INFO"

    def origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
