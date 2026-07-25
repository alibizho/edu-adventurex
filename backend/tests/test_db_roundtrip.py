"""Direct DbStore round-trip against a live Postgres — exercises every Store method and the
JSONB (de)serialization boundary. Not hermetic: needs a reachable Postgres at $DATABASE_URL with
STORE_BACKEND=db. Designed to run INSIDE the backend container (which is on the compose network):

    docker compose exec -T backend python - < backend/tests/test_db_roundtrip.py

It creates a throwaway session, verifies durability + upsert semantics for all tables, then
deletes the session (FK CASCADE cleans up children) so the DB is left as it was found.
"""
import asyncio
import os
import sys
import uuid

from sqlalchemy import text

from app.schemas import (
    Anomaly,
    ChunkAnalysis,
    ClassProgressRecord,
    CurriculumUpdate,
    QuestionDelta,
    RunResult,
    Score,
    Segment,
    SpeechProsody,
    StudentQuestion,
    TargetedQuestion,
    WordScore,
)
from app.store.db import DbStore


def check(cond: bool, msg: str) -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
    if not cond:
        raise AssertionError(msg)


async def main() -> None:
    url = os.environ.get("DATABASE_URL")
    if not url:
        print("DATABASE_URL not set; run inside the backend container.")
        sys.exit(2)
    print(f"Connecting to {url.rsplit('@', 1)[-1]}")

    st = DbStore(url)
    sid = f"dbtest-{uuid.uuid4().hex[:8]}"
    try:
        # -- init(): CREATE TABLE on a fresh schema is idempotent (already exists in a live stack) --
        await st.init()
        check(True, "init() created/verified tables")

        # -- session topic (also lazily creates the sessions row) --
        await st.set_topic(sid, "Photosynthesis")
        check(await st.get_topic(sid) == "Photosynthesis", "topic set/get")

        # -- transcript --
        await st.append_segment(sid, Segment(id=0, idx=0, text="Plants use light.", t_start=0.0, t_end=1.2))
        await st.append_segment(sid, Segment(id=1, idx=1, text="They make sugar."))
        tx = await st.get_transcript(sid)
        check([s.id for s in tx] == [0, 1], "transcript append/get ordered by seg_id")
        check(tx[0].t_start == 0.0 and tx[1].t_start is None, "nullable float columns round-trip")

        # -- analyses: append is an upsert on (session_id, chunk_id); JSONB nested fields --
        await st.append_analysis(sid, ChunkAnalysis(
            chunk_id=0, text="Plants use light.", confidence=0.9,
            anomalies=[Anomaly(type="hedging", source="mock", score=0.1, evidence="um")],
            detail=[WordScore(word="light", hesitation_zscore=1.5, is_anomaly=True)],
        ))
        await st.append_analysis(sid, ChunkAnalysis(
            chunk_id=1,
            text="They make sugar.",
            confidence=0.4,
            student_question=StudentQuestion(
                question_text="Where does the carbon come from?",
                target_concept="carbon fixation",
                anomaly_type="recall_failure",
            ),
            curriculum_update=CurriculumUpdate(
                added_concepts=["Calvin cycle"]
            ),
            # Set by fusion.fuse_prosody. These had no columns and were silently dropped, so the
            # GPU-vs-browser comparison gpu_confidence exists for was impossible against Postgres.
            prosody=SpeechProsody(
                speech_ms=4200, total_ms=6000, pause_count=3,
                longest_pause_ms=900, mean_level=0.031, peak_level=0.184,
            ),
            gpu_confidence=0.72,
        ))
        # re-analyze chunk 0 -> should UPDATE in place, not duplicate
        await st.append_analysis(sid, ChunkAnalysis(chunk_id=0, text="Plants use light.", confidence=0.55))
        an = await st.get_analyses(sid)
        check(len(an) == 2, "append_analysis upserts (no duplicate for re-analyzed chunk 0)")
        check(an[0].confidence == 0.55, "re-analyzed chunk 0 updated in place")
        check(an[1].anomalies == [] and an[0].chunk_id == 0, "analyses ordered by chunk_id")
        check(
            an[1].student_question is not None
            and an[1].student_question.target_concept == "carbon fixation",
            "GPU student question JSONB round-trip",
        )
        check(
            an[1].curriculum_update is not None
            and an[1].curriculum_update.added_concepts == ["Calvin cycle"],
            "curriculum update JSONB round-trip",
        )
        check(
            an[1].prosody is not None
            and an[1].prosody.pause_count == 3
            and an[1].prosody.longest_pause_ms == 900
            and abs(an[1].prosody.peak_level - 0.184) < 1e-6,
            "browser prosody JSONB round-trip",
        )
        check(an[1].gpu_confidence == 0.72, "pre-fusion gpu_confidence round-trip")
        # The upsert must carry them too, or a re-analyzed chunk keeps stale prosody.
        await st.append_analysis(sid, ChunkAnalysis(
            chunk_id=1, text="They make sugar.", confidence=0.4,
            prosody=SpeechProsody(speech_ms=1000, total_ms=1200, pause_count=0,
                                  longest_pause_ms=0, mean_level=0.05, peak_level=0.2),
            gpu_confidence=0.31,
        ))
        an = await st.get_analyses(sid)
        check(
            an[1].prosody is not None and an[1].prosody.pause_count == 0
            and an[1].gpu_confidence == 0.31,
            "re-analyzed chunk updates prosody + gpu_confidence in place",
        )

        # set_analyses replaces the whole set
        await st.set_analyses(sid, [ChunkAnalysis(chunk_id=5, text="replaced", confidence=0.7)])
        an = await st.get_analyses(sid)
        check([a.chunk_id for a in an] == [5], "set_analyses replaces all analyses")

        # -- measurement run + scores (JSONB lists of pydantic models) --
        run = RunResult(
            session_id=sid, delta_overall=0.42, survival_rate=0.75,
            per_question=[QuestionDelta(question_id=0, taught_mean=0.8, cold_mean=0.4, delta=0.4)],
            calibration_rho=0.61,
        )
        scores = [Score(question_id=0, arm="taught", persona_seed="p1", correct=True, cited_segment_ids=[0, 1])]
        await st.set_run(sid, run, scores)
        got = await st.get_run(sid)
        check(got is not None and got.delta_overall == 0.42, "run set/get round-trip")
        check(got.per_question[0].delta == 0.4 and got.calibration_rho == 0.61, "run JSONB nested fields")
        gs = await st.get_scores(sid)
        check(len(gs) == 1 and gs[0].cited_segment_ids == [0, 1], "scores JSONB round-trip")
        # re-run upserts on the session PK
        run.delta_overall = 0.99
        await st.set_run(sid, run, [])
        check((await st.get_run(sid)).delta_overall == 0.99, "set_run upserts on session_id")

        # -- targeted-question ledger --
        check(await st.next_question_id(sid) == 0, "next_question_id starts at 0")
        await st.record_questions(sid, [
            TargetedQuestion(id=0, chunk_id=5, text="Why sugar?", anomaly_type="hedging", rationale="low conf"),
            TargetedQuestion(id=1, chunk_id=5, text="Which wavelength?"),
        ])
        check(await st.next_question_id(sid) == 2, "next_question_id advances past recorded ids")
        hist = await st.get_history(sid)
        check(len(hist) == 2 and all(e.answer is None for e in hist), "questions recorded unanswered")
        check(await st.covered_chunk_ids(sid) == set(), "no covered chunks before any answer")

        ok = await st.record_answer(sid, 1, "Blue and red light.")
        check(ok is True, "record_answer returns True for a known question")
        check(await st.record_answer(sid, 99, "x") is False, "record_answer returns False for unknown id")
        hist = await st.get_history(sid)
        answered = {e.question.id: e for e in hist}
        check(answered[1].answer == "Blue and red light.", "answer persisted")
        check(answered[1].answered_at is not None, "answered_at timestamp set")
        check(await st.covered_chunk_ids(sid) == {5}, "answered chunk now counts as covered")

        # -- verifier-gated conversation: answer_key + parent_id survive, and the thread is walked --
        await st.record_questions(sid, [
            TargetedQuestion(id=2, chunk_id=6, text="What does a routing table hold?",
                             answer_key="Destination prefixes mapped to a next hop."),
            TargetedQuestion(id=3, chunk_id=7, text="And what picks the row?", parent_id=2),
        ])
        threaded = {e.question.id: e for e in await st.get_history(sid)}
        check(
            threaded[2].question.answer_key == "Destination prefixes mapped to a next hop.",
            "answer_key round-trips (the verifier has nothing to grade against without it)",
        )
        check(threaded[3].question.parent_id == 2, "parent_id round-trips")
        check(await st.find_question(sid, 2) is not None, "find_question locates a live question")
        check(await st.find_question(sid, 404) is None, "find_question returns None for unknown id")
        check(await st.thread_turns(sid, 3) == 0, "thread starts with no answered turns")
        await st.record_answer(sid, 2, "it maps destinations to a next hop")
        check(await st.thread_turns(sid, 3) == 1, "thread_turns counts the parent's answer")
        await st.record_answer(sid, 3, "the longest prefix match")
        check(await st.thread_turns(sid, 2) == 2, "thread_turns counts the whole chain either way")

        # -- path memory: the struggle ledger rides inside the PathMemory JSONB blob --
        pid = f"path-{sid}"
        fresh = await st.get_memory(pid)
        check(fresh.path_id == pid and fresh.class_progress == {}, "get_memory defaults for a new path")
        fresh.class_progress["c1"] = ClassProgressRecord(
            turn_count=3,
            struggle_scores={"routing table": 1.4, "hop": 0.5},
            focus_target="routing table",
            covered_objectives=["o1"],
        )
        await st.update_memory(pid, fresh)
        reloaded = await st.get_memory(pid)
        progress = reloaded.class_progress["c1"]
        check(progress.struggle_scores == {"routing table": 1.4, "hop": 0.5},
              "struggle ledger round-trips inside the PathMemory blob")
        check(progress.focus_target == "routing table", "focus_target round-trips")
        check(progress.covered_objectives == ["o1"], "covered objectives still round-trip")

        print("\nALL DB ROUND-TRIP CHECKS PASSED")
    finally:
        # cleanup: delete the session; FK ondelete=CASCADE removes segments/analyses/runs/qa_entries
        async with st._engine.begin() as conn:
            await conn.execute(text("DELETE FROM sessions WHERE session_id = :s"), {"s": sid})
        await st.dispose()
        print(f"cleaned up session {sid}")


if __name__ == "__main__":
    asyncio.run(main())
