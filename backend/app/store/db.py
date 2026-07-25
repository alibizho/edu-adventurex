import time

from sqlalchemy import delete, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from ..schemas import (
    AnalysisJob,
    Anomaly,
    ChunkAnalysis,
    ClassUnit,
    CurriculumUpdate,
    GrowthPath,
    PathMemory,
    QAEntry,
    QuestionDelta,
    RunResult,
    Score,
    Segment,
    SpeechProsody,
    StudentQuestion,
    TargetedQuestion,
    WordScore,
)
from .base import count_thread_answers
from .models import (
    AnalysisJobRow,
    AnalysisRow,
    Base,
    GrowthPathRow,
    PathMemoryRow,
    QAEntryRow,
    RunRow,
    SegmentRow,
    SessionRow,
)

def _analysis_values(session_id: str, a: ChunkAnalysis) -> dict:
    return {
        "session_id": session_id,
        "chunk_id": a.chunk_id,
        "text": a.text,
        "confidence": a.confidence,
        "localized_target": a.localized_target,
        "anomalies": [an.model_dump() for an in a.anomalies],
        "detail": [w.model_dump() for w in a.detail],
        "student_question": (
            a.student_question.model_dump() if a.student_question is not None else None
        ),
        "curriculum_update": (
            a.curriculum_update.model_dump() if a.curriculum_update is not None else None
        ),
        "prosody": a.prosody.model_dump() if a.prosody is not None else None,
        "gpu_confidence": a.gpu_confidence,
    }

def _row_to_analysis(row: AnalysisRow) -> ChunkAnalysis:
    return ChunkAnalysis(
        chunk_id=row.chunk_id,
        text=row.text,
        confidence=row.confidence,
        localized_target=row.localized_target,
        anomalies=[Anomaly.model_validate(a) for a in (row.anomalies or [])],
        detail=[WordScore.model_validate(w) for w in (row.detail or [])],
        student_question=(
            StudentQuestion.model_validate(row.student_question)
            if row.student_question
            else None
        ),
        curriculum_update=(
            CurriculumUpdate.model_validate(row.curriculum_update)
            if row.curriculum_update
            else None
        ),
        prosody=SpeechProsody.model_validate(row.prosody) if row.prosody else None,
        gpu_confidence=row.gpu_confidence,
    )

def _row_to_qa(row: QAEntryRow) -> QAEntry:
    return QAEntry(
        question=TargetedQuestion(
            id=row.question_id,
            chunk_id=row.chunk_id,
            text=row.text,
            anomaly_type=row.anomaly_type,
            rationale=row.rationale,
            answer_key=row.answer_key,
            parent_id=row.parent_id,
        ),
        answer=row.answer,
        answered_at=row.answered_at,
    )

class DbStore:

    def __init__(self, url: str):
        self._engine = create_async_engine(url, pool_pre_ping=True)
        self._sessionmaker: async_sessionmaker[AsyncSession] = async_sessionmaker(
            self._engine, expire_on_commit=False
        )

    async def init(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            await conn.execute(
                text("ALTER TABLE analyses ADD COLUMN IF NOT EXISTS student_question JSONB")
            )
            await conn.execute(
                text("ALTER TABLE analyses ADD COLUMN IF NOT EXISTS curriculum_update JSONB")
            )
            await conn.execute(
                text("ALTER TABLE analyses ADD COLUMN IF NOT EXISTS prosody JSONB")
            )
            await conn.execute(
                text(
                    "ALTER TABLE analyses ADD COLUMN IF NOT EXISTS "
                    "gpu_confidence DOUBLE PRECISION"
                )
            )
            await conn.execute(
                text("ALTER TABLE qa_entries ADD COLUMN IF NOT EXISTS answer_key TEXT")
            )
            await conn.execute(
                text("ALTER TABLE qa_entries ADD COLUMN IF NOT EXISTS parent_id INTEGER")
            )

    async def dispose(self) -> None:
        await self._engine.dispose()

    async def clear_session(self, session_id: str) -> None:
        async with self._sessionmaker() as s:
            await s.execute(delete(AnalysisJobRow).where(AnalysisJobRow.session_id == session_id))
            await s.execute(delete(SessionRow).where(SessionRow.session_id == session_id))
            await s.commit()

    async def _ensure_session(self, s: AsyncSession, session_id: str) -> None:
        await s.execute(
            pg_insert(SessionRow)
            .values(session_id=session_id)
            .on_conflict_do_nothing(index_elements=["session_id"])
        )

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
                        for k in (
                            "text",
                            "confidence",
                            "localized_target",
                            "anomalies",
                            "detail",
                            "student_question",
                            "curriculum_update",
                            "prosody",
                            "gpu_confidence",
                        )
                    },
                )
            )
            await s.commit()

    async def set_analyses(self, session_id: str, analyses: list[ChunkAnalysis]) -> None:
        async with self._sessionmaker() as s:
            await self._ensure_session(s, session_id)
            await s.execute(
                delete(AnalysisRow).where(AnalysisRow.session_id == session_id)
            )
            for a in analyses:
                s.add(AnalysisRow(**_analysis_values(session_id, a)))
            await s.commit()

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
                        answer_key=q.answer_key,
                        parent_id=q.parent_id,
                    )
                )
            await s.commit()

    async def find_question(self, session_id: str, question_id: int) -> QAEntry | None:
        async with self._sessionmaker() as s:
            row = (
                await s.execute(
                    select(QAEntryRow).where(
                        QAEntryRow.session_id == session_id,
                        QAEntryRow.question_id == question_id,
                    )
                )
            ).scalar_one_or_none()
            return _row_to_qa(row) if row else None

    async def thread_turns(self, session_id: str, question_id: int) -> int:
        return count_thread_answers(await self.get_history(session_id), question_id)

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

    async def save_path(self, path: GrowthPath) -> None:
        async with self._sessionmaker() as s:
            payload = path.model_dump()
            await s.execute(
                pg_insert(GrowthPathRow)
                .values(path_id=path.path_id, payload=payload)
                .on_conflict_do_update(index_elements=["path_id"], set_={"payload": payload})
            )
            await s.commit()

    async def get_path(self, path_id: str) -> GrowthPath | None:
        async with self._sessionmaker() as s:
            row = (
                await s.execute(
                    select(GrowthPathRow).where(GrowthPathRow.path_id == path_id)
                )
            ).scalar_one_or_none()
            return GrowthPath.model_validate(row.payload) if row else None

    async def list_paths(self) -> list[GrowthPath]:
        async with self._sessionmaker() as s:
            rows = (
                await s.execute(select(GrowthPathRow).order_by(GrowthPathRow.created_at.desc()))
            ).scalars().all()
            return [GrowthPath.model_validate(row.payload) for row in rows]

    async def save_class(self, path_id: str, cls: ClassUnit) -> None:
        path = await self.get_path(path_id)
        if path is None:
            return
        for i, c in enumerate(path.classes):
            if c.class_id == cls.class_id:
                path.classes[i] = cls
                break
        await self.save_path(path)

    async def get_memory(self, path_id: str) -> PathMemory:
        async with self._sessionmaker() as s:
            row = (
                await s.execute(
                    select(PathMemoryRow).where(PathMemoryRow.path_id == path_id)
                )
            ).scalar_one_or_none()
            return PathMemory.model_validate(row.payload) if row else PathMemory(path_id=path_id)

    async def update_memory(self, path_id: str, memory: PathMemory) -> None:
        async with self._sessionmaker() as s:
            payload = memory.model_dump()
            await s.execute(
                pg_insert(PathMemoryRow)
                .values(path_id=path_id, payload=payload)
                .on_conflict_do_update(index_elements=["path_id"], set_={"payload": payload})
            )
            await s.commit()

    async def get_analysis_job(self, session_id: str) -> AnalysisJob | None:
        async with self._sessionmaker() as s:
            row = (
                await s.execute(
                    select(AnalysisJobRow).where(AnalysisJobRow.session_id == session_id)
                )
            ).scalar_one_or_none()
            if row is None:
                return None
            return AnalysisJob(
                session_id=row.session_id,
                status=row.status,
                error=row.error,
                updated_at=row.updated_at.timestamp(),
            )

    async def set_analysis_job(self, job: AnalysisJob) -> None:
        from datetime import datetime, timezone

        updated_at = datetime.fromtimestamp(job.updated_at, timezone.utc)
        async with self._sessionmaker() as s:
            await self._ensure_session(s, job.session_id)
            await s.execute(
                pg_insert(AnalysisJobRow)
                .values(
                    session_id=job.session_id,
                    status=job.status,
                    error=job.error,
                    updated_at=updated_at,
                )
                .on_conflict_do_update(
                    index_elements=["session_id"],
                    set_={"status": job.status, "error": job.error, "updated_at": updated_at},
                )
            )
            await s.commit()
