"""Session storage. Features and scores only: no raw drawings, no raw strokes."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, Index, String, Text, create_engine, delete, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool

from backend.schemas import SessionCreate


class Base(DeclarativeBase):
    pass


class SessionRow(Base):
    __tablename__ = "sessions"
    __table_args__ = (Index("ix_sessions_device_created", "device_id", "created_at"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    device_id: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    kind: Mapped[str] = mapped_column(String(16))
    label: Mapped[str] = mapped_column(String(24))
    probability: Mapped[float] = mapped_column(Float)
    tremor_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    spiral_deviation: Mapped[float | None] = mapped_column(Float, nullable=True)
    pointer_type: Mapped[str] = mapped_column(String(16))
    model_version: Mapped[str] = mapped_column(String(64))
    features_json: Mapped[str] = mapped_column(Text)


def normalize_url(url: str) -> str:
    """Render and Heroku hand out postgres:// URLs; SQLAlchemy with psycopg 3
    needs the driver spelled out."""
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix) :]
    return url


def make_engine(url: str) -> Engine:
    url = normalize_url(url)
    if url.startswith("sqlite"):
        in_memory = ":memory:" in url or url in ("sqlite://", "sqlite:///")
        if in_memory:
            return create_engine(
                url, connect_args={"check_same_thread": False}, poolclass=StaticPool
            )
        return create_engine(url, connect_args={"check_same_thread": False})
    return create_engine(url, pool_pre_ping=True)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False)


def add_session(db: Session, payload: SessionCreate, keep_latest: int) -> SessionRow:
    row = SessionRow(
        id=str(uuid.uuid4()),
        device_id=payload.device_id,
        created_at=datetime.now(UTC),
        kind=payload.kind,
        label=payload.label,
        probability=payload.probability,
        tremor_score=payload.tremor_score,
        spiral_deviation=payload.spiral_deviation,
        pointer_type=payload.pointer_type,
        model_version=payload.model_version,
        features_json=json.dumps(payload.features, sort_keys=True),
    )
    db.add(row)
    db.flush()
    trim_old_sessions(db, payload.device_id, keep_latest)
    db.commit()
    return row


def trim_old_sessions(db: Session, device_id: str, keep_latest: int) -> None:
    """Cap rows per device so one client cannot grow the table without limit."""
    count = db.scalar(select(func.count()).where(SessionRow.device_id == device_id)) or 0
    excess = count - keep_latest
    if excess <= 0:
        return
    oldest = db.scalars(
        select(SessionRow.id)
        .where(SessionRow.device_id == device_id)
        .order_by(SessionRow.created_at)
        .limit(excess)
    ).all()
    db.execute(delete(SessionRow).where(SessionRow.id.in_(oldest)))


def list_sessions(db: Session, device_id: str) -> list[SessionRow]:
    return list(
        db.scalars(
            select(SessionRow)
            .where(SessionRow.device_id == device_id)
            .order_by(SessionRow.created_at)
        )
    )


def delete_sessions(db: Session, device_id: str) -> int:
    result = db.execute(delete(SessionRow).where(SessionRow.device_id == device_id))
    db.commit()
    return int(result.rowcount or 0)
