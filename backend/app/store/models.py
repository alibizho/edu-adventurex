"""SQLAlchemy ORM models for the session context store. Nested fields (anomalies, detail,
per_question, scores) are JSONB. The Pydantic models in schemas.py are the serialization boundary —
rows convert to/from them via .model_dump() / .model_validate() (see store/db.py)."""
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


class SessionRow(Base):
    __tablename__ = "sessions"

    session_id: Mapped[str] = mapped_column(String, primary_key=True)
    topic: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    # _now() is tz-aware (UTC); the column must be TIMESTAMP WITH TIME ZONE or asyncpg rejects it.
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class SegmentRow(Base):
    __tablename__ = "segments"
    __table_args__ = (UniqueConstraint("session_id", "seg_id", name="uq_segments_session_seg"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.session_id", ondelete="CASCADE"))
    seg_id: Mapped[int] = mapped_column(Integer)       # the Segment.id (stable pk within the session)
    idx: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    t_start: Mapped[float | None] = mapped_column(Float, nullable=True)
    t_end: Mapped[float | None] = mapped_column(Float, nullable=True)


class AnalysisRow(Base):
    __tablename__ = "analyses"
    __table_args__ = (UniqueConstraint("session_id", "chunk_id", name="uq_analyses_session_chunk"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.session_id", ondelete="CASCADE"))
    chunk_id: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text, default="")
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    localized_target: Mapped[str | None] = mapped_column(Text, nullable=True)
    anomalies: Mapped[list] = mapped_column(JSONB, default=list)   # list[Anomaly] as dicts
    detail: Mapped[list] = mapped_column(JSONB, default=list)      # list[WordScore] as dicts


class RunRow(Base):
    __tablename__ = "runs"

    session_id: Mapped[str] = mapped_column(
        String, ForeignKey("sessions.session_id", ondelete="CASCADE"), primary_key=True
    )
    delta_overall: Mapped[float] = mapped_column(Float, default=0.0)
    survival_rate: Mapped[float] = mapped_column(Float, default=0.0)
    calibration_rho: Mapped[float | None] = mapped_column(Float, nullable=True)
    per_question: Mapped[list] = mapped_column(JSONB, default=list)  # list[QuestionDelta] as dicts
    scores: Mapped[list] = mapped_column(JSONB, default=list)        # list[Score] as dicts
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class QAEntryRow(Base):
    __tablename__ = "qa_entries"
    __table_args__ = (UniqueConstraint("session_id", "question_id", name="uq_qa_session_question"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.session_id", ondelete="CASCADE"))
    question_id: Mapped[int] = mapped_column(Integer)
    chunk_id: Mapped[int] = mapped_column(Integer, default=0)
    text: Mapped[str] = mapped_column(Text)
    anomaly_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    answered_at: Mapped[float | None] = mapped_column(Float, nullable=True)
