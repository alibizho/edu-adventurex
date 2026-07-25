from __future__ import annotations

import asyncio
import re
import time
from collections import defaultdict

from ..agents.student import next_segment, student_turn
from ..agents.targeted import explain_answer, generate_targeted_questions
from ..config import settings
from ..confusion import engine
from ..pipeline.grading import grade_answer
from . import mastery
from ..schemas import (
    AudioClassTeachResponse,
    ChunkAnalysis,
    ClassObjective,
    ClassProgressRecord,
    ClassTeachResponse,
    ClassUnit,
    PathMemory,
    QAEntry,
    Segment,
    TargetedQuestion,
)
from ..store import store

def class_session_id(path_id: str, class_id: str) -> str:
    return f"{path_id}:{class_id}"

MEMORY_HISTORY_LIMIT = 20

def _memory_as_history(memory: PathMemory) -> list[QAEntry]:
    return [
        QAEntry(question=TargetedQuestion(id=-1, chunk_id=-1, text=q))
        for q in memory.asked_questions[-MEMORY_HISTORY_LIMIT:]
    ]

def _readiness(cls: ClassUnit, progress: ClassProgressRecord) -> int:
    checklist = cls.checklist()
    if not checklist:
        return 0
    covered = sum(1 for o in checklist if o.id in progress.covered_objectives)
    return round(100 * covered / len(checklist))

def _advance_progress(memory: PathMemory, class_id: str, cls: ClassUnit) -> ClassProgressRecord:
    now = time.time()
    progress = memory.class_progress.get(class_id) or ClassProgressRecord()
    if progress.started_at is None:
        progress.started_at = now
    progress.status = "in_progress"
    progress.turn_count += 1
    progress.readiness = _readiness(cls, progress)
    memory.class_progress[class_id] = progress
    return progress

def _apply_curriculum_update(memory: PathMemory, analysis: ChunkAnalysis) -> None:
    update = analysis.curriculum_update
    if update is None:
        return
    for concept in update.added_concepts:
        clean = concept.strip()
        if clean and clean not in memory.expanded_concepts:
            memory.expanded_concepts.append(clean)

STRUGGLE_ANOMALIES = frozenset({"factual_error", "fluency_issue", "recall_failure"})
MASTERY_CONFIDENCE = 0.85
MASTERY_DECAY = 0.5
MASTERED_BELOW = 0.2
LEDGER_LIMIT = 12

_FILLER = frozenset("""
a an the this that these those there here it its i me my we our you your he she they them his her
is are was were be am do does did have has had will would can could may might must of in on at to
from by for with about and or but so because if then than as not no nor yes yeah yep nope ok okay
um uh erm hmm ah oh eh like just really very quite too also more most some any all now then when
what which who why how well right sure thanks thank please sorry actually basically kind sort lot
bro dude guys guy man see say said says get got go know think want need let
usually often always never sometimes maybe probably good bad nice cool great fine better best
different same other another such thing things stuff way ways bit part yet still even much many
one two three first second next last
it's its that's there's here's i'm i've you're we're they're he's she's don't doesn't didn't isn't
aren't wasn't can't won't wouldn't couldn't shouldn't gonna wanna gotta
""".split())

_HEDGES = frozenset(
    "essentially obviously honestly frankly apparently totally kinda sorta".split())
_FILLER_BASES = frozenset(
    "um uh hm er erm ah oh oo oof ooh ugh huh mm m so ha he aw eh ow o a e u".split())
_FILLER_RE = re.compile(r"(.)\1+")

def _is_filler_token(token: str) -> bool:
    clean = token.strip().strip(".,!?;:\"'()[]{}…-–—").casefold()
    if not clean:
        return True
    collapsed = _FILLER_RE.sub(r"\1", clean)
    return (clean in _FILLER or clean in _HEDGES
            or collapsed in _FILLER_BASES or collapsed in _HEDGES)

def _is_content_token(token: str) -> bool:
    clean = token.strip().strip(".,!?;:\"'()[]{}…-–—").casefold()
    return len(clean) > 2 and not clean.isnumeric() and not _is_filler_token(clean)

def _strip_filler(target: str) -> str:
    if not target:
        return ""
    return " ".join(t for t in target.split() if _is_content_token(t)).strip()

def is_concept_like(target: str) -> bool:
    return bool(_strip_filler(target))

_GIVE_UP = (
    "i don't know", "i dont know", "i do not know", "dunno", "idk", "no idea", "not sure",
    "i'm not sure", "im not sure", "no clue", "i give up", "give up", "you tell me", "just tell me",
    "tell me the answer", "explain it to me", "i forgot", "i can't remember", "i cant remember",
    "i can't explain", "i cant explain", "skip", "pass", "next question",
    "i don't understand", "i dont understand", "i don't get it", "i dont get it",
    "i don't remember", "i dont remember", "not a clue", "beats me", "what's the answer",
    "whats the answer", "can you tell me", "i wish i knew",
)

_GIVE_UP_PATTERN = re.compile(
    r"\bi(?:'m|m| am)\s+"
    r"(?:really |totally |completely |so |pretty |quite |honestly |kind of |kinda |a bit )?"
    r"(?:lost|stuck|blanking|clueless|drawing a blank)\b"
)

_GIVE_UP_LEAD_WORDS = 3

def _give_up_at(clean: str) -> int | None:
    hits = [at for at in (clean.find(phrase) for phrase in _GIVE_UP) if at >= 0]
    found = _GIVE_UP_PATTERN.search(clean)
    if found:
        hits.append(found.start())
    return min(hits) if hits else None

def _is_give_up(answer: str) -> bool:
    clean = answer.strip().casefold().replace("’", "'").replace("ʼ", "'")
    if not _strip_filler(clean):
        return True
    at = _give_up_at(clean)
    return at is not None and len(clean[:at].split()) <= _GIVE_UP_LEAD_WORDS

def _mentions(text: str, concept: str) -> bool:
    concept = concept.strip().casefold()
    return bool(concept) and concept in text.casefold()

def _decay_concepts(progress: ClassProgressRecord, *texts: str) -> None:
    for concept in list(progress.struggle_scores):
        if any(_mentions(text, concept) for text in texts):
            progress.struggle_scores[concept] = round(
                progress.struggle_scores[concept] * MASTERY_DECAY, 4
            )
            if progress.struggle_scores[concept] < MASTERED_BELOW:
                del progress.struggle_scores[concept]
    progress.focus_target = (
        max(progress.struggle_scores, key=lambda k: progress.struggle_scores[k])
        if progress.struggle_scores else ""
    )

def _update_struggle_ledger(progress: ClassProgressRecord, analysis: ChunkAnalysis) -> None:
    scores = progress.struggle_scores
    target = (analysis.localized_target or "").strip()
    if not is_concept_like(target):
        target = ""

    if target:
        struggle = sum(a.score for a in analysis.anomalies if a.type in STRUGGLE_ANOMALIES)
        if struggle > 0:
            scores[target] = round(scores.get(target, 0.0) + struggle, 4)

    if analysis.confidence > MASTERY_CONFIDENCE and not analysis.anomalies:
        _decay_concepts(progress, analysis.text)

    if len(scores) > LEDGER_LIMIT:
        keep = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:LEDGER_LIMIT]
        progress.struggle_scores = dict(keep)
        scores = progress.struggle_scores

    progress.focus_target = max(scores, key=lambda k: scores[k]) if scores else ""

def _gpu_asked(analysis: ChunkAnalysis) -> bool:
    generated = analysis.student_question
    return (
        generated is not None
        and bool(generated.question_text.strip())
        and is_concept_like(generated.target_concept)
    )

def _gpu_hint(analysis: ChunkAnalysis) -> str:
    generated = analysis.student_question
    candidate = (generated.target_concept if generated else "") or (
        analysis.localized_target or ""
    )
    return _strip_filler(candidate)

async def _question_from_analysis(
    session_id: str,
    analysis: ChunkAnalysis,
    memory: PathMemory,
) -> TargetedQuestion | None:
    generated = analysis.student_question
    if generated is None or not generated.question_text.strip():
        return None
    if not is_concept_like(generated.target_concept):
        return None
    text = generated.question_text.strip()
    if any(existing.casefold() == text.casefold() for existing in memory.asked_questions):
        return None
    return TargetedQuestion(
        id=await store.next_question_id(session_id),
        chunk_id=analysis.chunk_id,
        text=text,
        anomaly_type=generated.anomaly_type,
        rationale=f"GPU confusion signal on {generated.target_concept}",
    )

_memory_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

async def _save_memory(
    path_id: str, class_id: str, cls: ClassUnit, memory: PathMemory
) -> None:
    async with _memory_locks[path_id]:
        stored = (await store.get_memory(path_id)).class_progress.get(class_id)
        progress = memory.class_progress.get(class_id)
        if stored is not None and progress is not None and stored.reset_count != progress.reset_count:
            return
        if stored is not None and progress is not None:
            for objective_id in stored.covered_objectives:
                if objective_id not in progress.covered_objectives:
                    progress.covered_objectives.append(objective_id)
            progress.objective_evidence = {
                **stored.objective_evidence, **progress.objective_evidence
            }
            progress.last_checked_segment = max(
                progress.last_checked_segment, stored.last_checked_segment
            )
            progress.readiness = _readiness(cls, progress)
        await store.update_memory(path_id, memory)

async def class_teach_turn(
    path_id: str, class_id: str, cls: ClassUnit, latest_utterance: str
) -> ClassTeachResponse:
    session_id = class_session_id(path_id, class_id)

    transcript = await store.get_transcript(session_id)
    resp = await student_turn(transcript, latest_utterance)
    await store.append_segment(session_id, resp.new_segment)

    memory = await store.get_memory(path_id)
    progress = _advance_progress(memory, class_id, cls)

    open_objectives = [o for o in cls.checklist() if o.id not in progress.covered_objectives]
    if _is_give_up(latest_utterance) and (progress.focus_target or open_objectives):
        stuck = progress.focus_target or _next_objective(open_objectives, progress.focus_target).text
        said = await explain_answer(
            TargetedQuestion(id=-1, chunk_id=resp.new_segment.id, text=f"Explain {stuck}."),
            topic=f"{cls.title} — {cls.objective}",
            transcript=mastery.transcript_excerpt(transcript + [resp.new_segment]),
            objectives=[o.text for o in cls.checklist()],
        )
        progress.explanations_given += 1
        _decay_concepts(progress, stuck)
        await store.update_memory(path_id, memory)
        return ClassTeachResponse(
            student_reply=said, new_segment=resp.new_segment, asked=False, explained=True
        )

    analysis = engine.analyze([latest_utterance])[0]
    analysis.chunk_id = resp.new_segment.id
    if not engine.is_confused(analysis):
        await store.update_memory(path_id, memory)
        return ClassTeachResponse(
            student_reply=resp.student_reply, new_segment=resp.new_segment, asked=False
        )

    history = await store.get_history(session_id) + _memory_as_history(memory)
    questions = await generate_targeted_questions(
        [analysis],
        history,
        start_id=await store.next_question_id(session_id),
        topic=f"{cls.title} — {cls.objective}",
        transcript=mastery.transcript_excerpt(transcript + [resp.new_segment]),
        focus_target=progress.focus_target,
        objectives=[o.text for o in cls.checklist()],
    )
    if not questions:
        await store.update_memory(path_id, memory)
        return ClassTeachResponse(
            student_reply=resp.student_reply, new_segment=resp.new_segment, asked=False
        )

    await store.record_questions(session_id, questions)
    memory.asked_questions.append(questions[0].text)
    await store.update_memory(path_id, memory)
    return ClassTeachResponse(
        student_reply=resp.student_reply,
        new_segment=resp.new_segment,
        asked=True,
        question=questions[0],
    )

def _probe_is_due(progress: ClassProgressRecord) -> bool:
    return progress.turn_count - progress.last_goal_probe_turn >= settings.goal_probe_cooldown

def _next_objective(
    open_objectives: list[ClassObjective], focus_target: str
) -> ClassObjective:
    focus = focus_target.strip().casefold()
    if focus:
        for objective in open_objectives:
            if focus in objective.text.casefold():
                return objective
    return open_objectives[0]

async def _goal_probe_question(
    session_id: str,
    cls: ClassUnit,
    objective: ClassObjective,
    transcript: list[Segment],
    memory: PathMemory,
) -> TargetedQuestion | None:
    text = await mastery.goal_probe(cls, objective, mastery.transcript_excerpt(transcript))
    if not text or text in memory.asked_questions:
        return None
    return TargetedQuestion(
        id=await store.next_question_id(session_id),
        chunk_id=transcript[-1].id if transcript else 0,
        text=text,
        anomaly_type="uncovered_goal",
        rationale=f"objective not yet covered: {objective.text}",
    )

MAX_CONVERSATION_TURNS = 3

UNGRADED_CONVERSATION_TURNS = 2

async def _conversation_turn(
    session_id: str,
    cls: ClassUnit,
    answering: QAEntry,
    answer: str,
    analysis: ChunkAnalysis,
    memory: PathMemory,
    progress: ClassProgressRecord,
    transcript: list[Segment],
) -> tuple[str, TargetedQuestion | None, bool | None, bool]:
    question = answering.question
    turns = await store.thread_turns(session_id, question.id)
    objectives = [o.text for o in cls.checklist()]
    topic = f"{cls.title} — {cls.objective}"

    gave_up = _is_give_up(answer)
    correct, _not_covered = (False, False) if gave_up else await grade_answer(answer, question)
    gradable = bool((question.answer_key or "").strip())
    cap = MAX_CONVERSATION_TURNS if gradable else UNGRADED_CONVERSATION_TURNS

    if correct:
        _decay_concepts(progress, question.text, answer)
        return "OH — THAT MAKES SENSE NOW. THANKS!", None, True, True

    if gave_up or turns >= cap:
        said = await explain_answer(
            question,
            topic=topic,
            transcript=mastery.transcript_excerpt(transcript),
            objectives=objectives,
        )
        progress.explanations_given += 1
        _decay_concepts(progress, question.text)
        return said, None, (None if not gradable else False), True

    follow_ups = await generate_targeted_questions(
        [analysis],
        await store.get_history(session_id) + _memory_as_history(memory),
        start_id=await store.next_question_id(session_id),
        topic=topic,
        transcript=mastery.transcript_excerpt(transcript),
        focus_target=progress.focus_target or question.text,
        parent_id=question.id,
        objectives=objectives,
    )
    if not follow_ups:
        return "OKAY, LET ME THINK ABOUT THAT.", None, False, True
    follow_up = follow_ups[0]
    return follow_up.text, follow_up, False, False

async def class_audio_turn(
    path_id: str,
    class_id: str,
    cls: ClassUnit,
    analysis: ChunkAnalysis,
    degraded: bool,
    silent: bool = False,
    answering_question_id: int | None = None,
) -> AudioClassTeachResponse:
    if degraded or not analysis.text.strip():
        return AudioClassTeachResponse(analysis=analysis, degraded=True)

    session_id = class_session_id(path_id, class_id)
    prior_transcript = await store.get_transcript(session_id)
    segment = next_segment(prior_transcript, analysis.text.strip())
    analysis.chunk_id = segment.id
    await store.append_segment(session_id, segment)
    transcript_with_latest = prior_transcript + [segment]
    await store.append_analysis(session_id, analysis)

    memory = await store.get_memory(path_id)
    progress = _advance_progress(memory, class_id, cls)
    _apply_curriculum_update(memory, analysis)
    _update_struggle_ledger(progress, analysis)

    if answering_question_id is not None:
        answering = await store.find_question(session_id, answering_question_id)
        if answering is not None:
            answer = analysis.text.strip()
            await store.record_answer(session_id, answering_question_id, answer)
            reply, follow_up, correct, over = await _conversation_turn(
                session_id, cls, answering, answer, analysis, memory, progress,
                transcript_with_latest,
            )
            if follow_up is not None:
                await store.record_questions(session_id, [follow_up])
                if follow_up.text not in memory.asked_questions:
                    memory.asked_questions.append(follow_up.text)
            await _save_memory(path_id, class_id, cls, memory)
            return AudioClassTeachResponse(
                student_reply=reply,
                new_segment=segment,
                analysis=analysis,
                asked=follow_up is not None,
                question=follow_up,
                degraded=False,
                all_goals_covered=not [
                    o for o in cls.checklist() if o.id not in progress.covered_objectives
                ],
                answer_correct=correct,
                conversation_over=over,
                turns_used=await store.thread_turns(session_id, answering_question_id),
            )

    open_objectives = [o for o in cls.checklist() if o.id not in progress.covered_objectives]
    objectives = [o.text for o in cls.checklist()]
    excerpt = mastery.transcript_excerpt(transcript_with_latest)

    if _is_give_up(analysis.text) and (progress.focus_target or open_objectives):
        stuck = progress.focus_target or _next_objective(
            open_objectives, progress.focus_target
        ).text
        said = await explain_answer(
            TargetedQuestion(id=-1, chunk_id=segment.id, text=f"Explain {stuck}."),
            topic=f"{cls.title} — {cls.objective}",
            transcript=excerpt,
            objectives=objectives,
        )
        progress.explanations_given += 1
        _decay_concepts(progress, stuck)
        await _save_memory(path_id, class_id, cls, memory)
        return AudioClassTeachResponse(
            student_reply=said, new_segment=segment, analysis=analysis, degraded=False,
            explained=True, all_goals_covered=not open_objectives,
        )

    question: TargetedQuestion | None = None
    if engine.is_confused(analysis) or _gpu_asked(analysis):
        questions = await generate_targeted_questions(
            [analysis],
            await store.get_history(session_id) + _memory_as_history(memory),
            start_id=await store.next_question_id(session_id),
            topic=f"{cls.title} — {cls.objective}",
            transcript=excerpt,
            focus_target=progress.focus_target or _gpu_hint(analysis),
            objectives=objectives,
        )
        if questions:
            question = questions[0]
        else:
            question = await _question_from_analysis(session_id, analysis, memory)

    if question is None and open_objectives and _probe_is_due(progress):
        question = await _goal_probe_question(
            session_id, cls, _next_objective(open_objectives, progress.focus_target),
            transcript_with_latest, memory,
        )
        if question is not None:
            progress.last_goal_probe_turn = progress.turn_count

    if question is not None:
        await store.record_questions(session_id, [question])
        if question.text not in memory.asked_questions:
            memory.asked_questions.append(question.text)
        student_reply = question.text
    elif silent:
        student_reply = ""
    else:
        student_reply = (await student_turn(prior_transcript, analysis.text.strip())).student_reply

    await _save_memory(path_id, class_id, cls, memory)
    return AudioClassTeachResponse(
        student_reply=student_reply,
        new_segment=segment,
        analysis=analysis,
        asked=question is not None,
        question=question,
        degraded=False,
        all_goals_covered=not open_objectives,
    )

async def end_class(
    path_id: str,
    class_id: str,
    cls: ClassUnit,
    completion_mode: str = "self-teaching",
) -> PathMemory:
    session_id = class_session_id(path_id, class_id)
    memory = await store.get_memory(path_id)

    progress = memory.class_progress.get(class_id) or ClassProgressRecord()
    if progress.status == "complete":
        return memory
    now = time.time()
    progress.status = "complete"
    progress.readiness = _readiness(cls, progress)
    progress.passed_on_mastery = all(
        o.id in progress.covered_objectives for o in cls.checklist()
    )
    progress.started_at = progress.started_at or now
    progress.completed_at = now
    progress.completion_mode = (
        "guided-explanation"
        if completion_mode == "guided-explanation" or progress.explanations_given > 0
        else "self-teaching"
    )
    memory.class_progress[class_id] = progress

    if cls.title not in memory.covered_concepts:
        memory.covered_concepts.append(cls.title)

    for entry in await store.get_history(session_id):
        label = entry.question.text
        if entry.answer and not _is_give_up(entry.answer):
            if label not in memory.understood:
                memory.understood.append(label)
            if label in memory.struggled:
                memory.struggled.remove(label)
        elif label not in memory.struggled and label not in memory.understood:
            memory.struggled.append(label)

    for objective in cls.checklist():
        target = memory.understood if objective.id in progress.covered_objectives else memory.struggled
        if objective.text not in target:
            target.append(objective.text)

    await store.update_memory(path_id, memory)
    return memory

async def reset_class(path_id: str, class_id: str, cls: ClassUnit) -> PathMemory:
    session_id = class_session_id(path_id, class_id)
    async with _memory_locks[path_id]:
        history = await store.get_history(session_id)
        own = {e.question.text for e in history} | {o.text for o in cls.checklist()}
        memory = await store.get_memory(path_id)
        previous = memory.class_progress.get(class_id)
        memory.class_progress[class_id] = ClassProgressRecord(
            reset_count=(previous.reset_count + 1) if previous else 1
        )
        memory.asked_questions = [q for q in memory.asked_questions if q not in own]
        memory.understood = [c for c in memory.understood if c not in own]
        memory.struggled = [c for c in memory.struggled if c not in own]
        memory.covered_concepts = [c for c in memory.covered_concepts if c != cls.title]
        await store.update_memory(path_id, memory)
        await store.clear_session(session_id)
    return memory

COVERAGE_WINDOW_SEGMENTS = 6

async def run_objective_check(path_id: str, class_id: str, cls: ClassUnit) -> None:
    try:
        session_id = class_session_id(path_id, class_id)
        transcript = await store.get_transcript(session_id)
        if not transcript:
            return

        memory = await store.get_memory(path_id)
        progress = memory.class_progress.get(class_id) or ClassProgressRecord()
        judged_reset_count = progress.reset_count
        fresh = [s for s in transcript if s.id > progress.last_checked_segment]
        if not mastery.should_check(len(fresh)):
            return

        open_objectives = [o for o in cls.checklist() if o.id not in progress.covered_objectives]
        latest_segment = max(s.id for s in transcript)
        covered: dict[str, str] = {}
        if open_objectives:
            covered = await mastery.judge_coverage(
                cls,
                open_objectives,
                mastery.transcript_excerpt(transcript[-COVERAGE_WINDOW_SEGMENTS:]),
            )

        async with _memory_locks[path_id]:
            memory = await store.get_memory(path_id)
            progress = memory.class_progress.get(class_id) or ClassProgressRecord()
            if progress.reset_count != judged_reset_count:
                return
            for objective_id, evidence in covered.items():
                if objective_id not in progress.covered_objectives:
                    progress.covered_objectives.append(objective_id)
                progress.objective_evidence[objective_id] = evidence
            progress.last_checked_segment = max(progress.last_checked_segment, latest_segment)
            progress.readiness = _readiness(cls, progress)
            memory.class_progress[class_id] = progress
            await store.update_memory(path_id, memory)
    except Exception as exc:
        print(f"[mastery] objective check failed for {path_id}:{class_id} ({exc!r})")
