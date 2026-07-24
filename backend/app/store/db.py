"""Postgres-backed Store (async SQLAlchemy 2.0 + asyncpg). Durable across restarts. Selected when
STORE_BACKEND=db and DATABASE_URL are set. Nested fields (anomalies, detail, per_question, scores)
are JSONB; the schemas.py Pydantic models are the serialization boundary via .model_dump() /
.model_validate(). Upserts use PostgreSQL ON CONFLICT so re-analyzing a chunk or re-measuring a
session updates in place.
"""
import time

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from ..schemas import (
    Anomaly,
    ChunkAnalysis,
    QAEntry,
    QuestionDelta,
    RunResult,
    Score,
    Segment,
    TargetedQuestion,
    WordScore,
)
from .models import AnalysisRow, Base, QAEntryRow, RunRow, SegmentRow, SessionRow


def _analysis_values(session_id: str, a: ChunkAnalysis) -> dict:
    return {
        "session_id": session_id,
        "chunk_id": a.chunk_id,
        "text": a.text,
        "confidence": a.confidence,
        "localized_target": a.localized_target,
        "anomalies": [an.model_dump() for an in a.anomalies],
        "detail": [w.model_dump() for w in a.detail],
    }


def _row_to_analysis(row: AnalysisRow) -> ChunkAnalysis:
    return ChunkAnalysis(
        chunk_id=row.chunk_id,
        text=row.text,
        confidence=row.confidence,
        localized_target=row.localized_target,
        anomalies=[Anomaly.model_validate(a) for a in (row.anomalies or [])],
        detail=[WordScore.model_validate(w) for w in (row.detail or [])],
    )


def _row_to_qa(row: QAEntryRow) -> QAEntry:
    return QAEntry(
        question=TargetedQuestion(
            id=row.question_id,
            chunk_id=row.chunk_id,
            text=row.text,
            anomaly_type=row.anomaly_type,
            rationale=row.rationale,
        ),
        answer=row.answer,
        answered_at=row.answered_at,
    )


class DbStore:
    """Store impl backed by Postgres. Each method opens its own short session (fine for the
    hackathon's per-request granularity; the async engine pools connections underneath)."""

    def __init__(self, url: str):
        self._engine = create_async_engine(url, pool_pre_ping=True)
        self._sessionmaker: async_sessionmaker[AsyncSession] = async_sessionmaker(
            self._engine, expire_on_commit=False
        )

    async def init(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def dispose(self) -> None:
        await self._engine.dispose()

    async def _ensure_session(self, s: AsyncSession, session_id: str) -> None:
        """Insert a sessions row if absent (FK target for everything else). Race-safe upsert."""
        await s.execute(
            pg_insert(SessionRow)
            .values(session_id=session_id)
            .on_conflict_do_nothing(index_elements=["session_id"])
        )

    # ---- transcript ----

    async def get_transcript(self, session_id: str) -> list[Segment]:
        async with self._sessionmaker() as s:
            rows = (
                await s.execute(
                    select(SegmentRow)
                    .where(SegmentRow.session_id == session_id)
                    .order_by(SegmentRow.seg_id)
                )
            ).scalars().all()
            return [
                Segment(id=r.seg_id, idx=r.idx, text=r.text, t_start=r.t_start, t_end=r.t_end)
                for r in rows
            ]

    async def append_segment(self, session_id: str, segment: Segment) -> None:
        async with self._sessionmaker() as s:
            await self._ensure_session(s, session_id)
            s.add(
                SegmentRow(
                    session_id=session_id,
                    seg_id=segment.id,
                    idx=segment.idx,
                    text=segment.text,
                    t_start=segment.t_start,
                    t_end=segment.t_end,
                )
            )
            await s.commit()

    # ---- confusion analyses ----

    async def get_analyses(self, session_id: str) -> list[ChunkAnalysis]:
        async with self._sessionmaker() as s:
            rows = (
                await s.execute(
                    select(AnalysisRow)
                    .where(AnalysisRow.session_id == session_id)
                    .order_by(AnalysisRow.chunk_id)
                )
            ).scalars().all()
            return [_row_to_analysis(r) for r in rows]

    async def append_analysis(self, session_id: str, analysis: ChunkAnalysis) -> None:
        """Upsert by (session_id, chunk_id): a re-analyzed utterance updates in place."""
        async with self._sessionmaker() as s:
            await self._ensure_session(s, session_id)
            vals = _analysis_values(session_id, analysis)
            await s.execute(
                pg_insert(AnalysisRow)
                .values(**vals)
                .on_conflict_do_update(
                    index_elements=["session_id", "chunk_id"],
                    set_={
                        k: vals[k]
                        for k in ("text", "confidence", "localized_target", "anomalies", "detail")
                    },
                )
            )
            await s.commit()

    async def set_analyses(self, session_id: str, analyses: list[ChunkAnalysis]) -> None:
        """Replace all analyses for the session (used by /confusion/ingest and /confusion/mock)."""
        async with self._sessionmaker() as s:
            await self._ensure_session(s, session_id)
            await s.execute(
                delete(AnalysisRow).where(AnalysisRow.session_id == session_id)
            )
            for a in analyses:
                s.add(AnalysisRow(**_analysis_values(session_id, a)))
            await s.commit()

    # ---- measurement run + raw scores ----

    async def set_run(self, session_id: str, result: RunResult, scores: list[Score]) -> None:
        vals = {
            "session_id": session_id,
            "delta_overall": result.delta_overall,
            "survival_rate": result.survival_rate,
            "calibration_rho": result.calibration_rho,
            "per_question": [q.model_dump() for q in result.per_question],
            "scores": [sc.model_dump() for sc in scores],
        }
        async with self._sessionmaker() as s:
            await self._ensure_session(s, session_id)
            await s.execute(
                pg_insert(RunRow)
                .values(**vals)
                .on_conflict_do_update(
                    index_elements=["session_id"],
                    set_={
                        k: vals[k]
                        for k in (
                            "delta_overall",
                            "survival_rate",
                            "calibration_rho",
                            "per_question",
                            "scores",
                        )
                    },
                )
            )
            await s.commit()

    async def get_run(self, session_id: str) -> RunResult | None:
        async with self._sessionmaker() as s:
            row = (
                await s.execute(select(RunRow).where(RunRow.session_id == session_id))
            ).scalar_one_or_none()
            if row is None:
                return None
            return RunResult(
                session_id=row.session_id,
                delta_overall=row.delta_overall,
                survival_rate=row.survival_rate,
                per_question=[
                    QuestionDelta.model_validate(q) for q in (row.per_question or [])
                ],
                calibration_rho=row.calibration_rho,
            )

    async def get_scores(self, session_id: str) -> list[Score]:
        async with self._sessionmaker() as s:
            row = (
                await s.execute(select(RunRow).where(RunRow.session_id == session_id))
            ).scalar_one_or_none()
            if row is None:
                return []
            return [Score.model_validate(sc) for sc in (row.scores or [])]

    # ---- targeted-question ledger ----

    async def get_history(self, session_id: str) -> list[QAEntry]:
        async with self._sessionmaker() as s:
            rows = (
                await s.execute(
                    select(QAEntryRow)
                    .where(QAEntryRow.session_id == session_id)
                    .order_by(QAEntryRow.question_id)
                )
            ).scalars().all()
            return [_row_to_qa(r) for r in rows]

    async def next_question_id(self, session_id: str) -> int:
        async with self._sessionmaker() as s:
            ids = (
                await s.execute(
                    select(QAEntryRow.question_id).where(QAEntryRow.session_id == session_id)
                )
            ).scalars().all()
            return (max(ids) if ids else -1) + 1

    async def record_questions(self, session_id: str, questions: list[TargetedQuestion]) -> None:
        async with self._sessionmaker() as s:
            await self._ensure_session(s, session_id)
            for q in questions:
                s.add(
                    QAEntryRow(
                        session_id=session_id,
                        question_id=q.id,
                        chunk_id=q.chunk_id,
                        text=q.text,
                        anomaly_type=q.anomaly_type,
                        rationale=q.rationale,
                    )
                )
            await s.commit()

    async def record_answer(self, session_id: str, question_id: int, answer: str) -> bool:
        async with self._sessionmaker() as s:
            row = (
                await s.execute(
                    select(QAEntryRow).where(
                        QAEntryRow.session_id == session_id,
                        QAEntryRow.question_id == question_id,
                    )
                )
            ).scalar_one_or_none()
            if row is None:
                return False
            row.answer = answer
            row.answered_at = time.time()
            await s.commit()
            return True

    async def covered_chunk_ids(self, session_id: str) -> set[int]:
        async with self._sessionmaker() as s:
            ids = (
                await s.execute(
                    select(QAEntryRow.chunk_id).where(
                        QAEntryRow.session_id == session_id,
                        QAEntryRow.answer.is_not(None),
                    )
                )
            ).scalars().all()
            return set(ids)

    # ---- session topic ----

    async def set_topic(self, session_id: str, topic: str) -> None:
        async with self._sessionmaker() as s:
            await s.execute(
                pg_insert(SessionRow)
                .values(session_id=session_id, topic=topic)
                .on_conflict_do_update(
                    index_elements=["session_id"], set_={"topic": topic}
                )
            )
            await s.commit()

    async def get_topic(self, session_id: str) -> str:
        async with self._sessionmaker() as s:
            topic = (
                await s.execute(
                    select(SessionRow.topic).where(SessionRow.session_id == session_id)
                )
            ).scalar_one_or_none()
            return topic or ""
