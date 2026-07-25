"""Teaching a class = the existing pipeline, reused, plus cross-class memory.

The learner teaches a class to the in-character AI student (`student_turn`). We run each utterance
through the confusion mock (`engine`) and, ONLY when it sounds unconfident, generate one targeted
question (`generate_targeted_questions`). Cross-class memory (`PathMemory`) is injected as extra
"already asked" history so the AI doesn't repeat near-duplicate questions in later classes — it
only asks a fresh angle when the learner keeps sounding unsure.

Each class runs on its own store session: session_id = "<path_id>:<class_id>".
"""
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


# How many cross-class questions to carry into the "don't repeat" block. Unbounded, this grew with
# every class in the path and eventually dominated the prompt — a long blocklist and one sentence
# of actual context pushes the generator toward whatever it hasn't asked yet, which reads as random.
MEMORY_HISTORY_LIMIT = 20


def _memory_as_history(memory: PathMemory) -> list[QAEntry]:
    """Render recent cross-class asked questions as synthetic (prompt-only) history entries so the
    'do NOT repeat any of these' rule in generate_targeted_questions spans every class."""
    return [
        QAEntry(question=TargetedQuestion(id=-1, chunk_id=-1, text=q))
        for q in memory.asked_questions[-MEMORY_HISTORY_LIMIT:]
    ]


def _readiness(cls: ClassUnit, progress: ClassProgressRecord) -> int:
    """Share of this class's objectives actually covered. Note what this is NOT: talking for longer
    no longer moves it. A learner who says nothing of substance for ten turns stays at 0."""
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


# Anomaly types that mean "the learner struggled with this concept". cognitive_load is excluded:
# slow delivery alone is effort, not misunderstanding, and it fires on almost every long sentence.
STRUGGLE_ANOMALIES = frozenset({"factual_error", "fluency_issue", "recall_failure"})
MASTERY_CONFIDENCE = 0.85     # spoke this clearly, with nothing flagged -> credit toward mastery
MASTERY_DECAY = 0.5           # each clean mention halves the outstanding struggle
MASTERED_BELOW = 0.2          # below this the concept is considered learned and leaves the ledger
LEDGER_LIMIT = 12             # cap: the whole PathMemory is one JSONB blob, keep it bounded


# Mirrors the ml-service's content-word filter. Duplicated on purpose rather than shared: the two
# services deploy separately, and a stale GPU box must not be able to poison the ledger with
# filler words ("yeah", "bro", "the") that then become the question the student asks.
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


# --- filler firewall (mirrors ml-service/engine.py: is_filler / _strip_filler) ----------------
# The two services deploy separately and share no package, so this is duplicated on purpose: a
# stale GPU box must not be able to poison the struggle ledger with a filler, and the backend must
# still refuse one if the GPU's own firewall ever regresses. Keep this in sync with engine.py;
# tests/test_backend.py::test_filler_words_never_become_the_concept_being_chased is the shared
# regression guard for both copies.
_HEDGES = frozenset(
    "essentially obviously honestly frankly apparently totally kinda sorta".split())
# Bases of pure interjections that, when drawn out, repeat a letter ("umm"->"um", "sooo"->"so",
# "hmmm"->"hm"). Curated, NOT derived from _FILLER, so collapsing a real doubled letter ("bee"->
# "be", "inn"->"in", "book"->"bok") can never turn a content word into a false filler: only the
# collapsed form is matched against this set, never against the big _FILLER list.
_FILLER_BASES = frozenset(
    "um uh hm er erm ah oh oo oof ooh ugh huh mm m so ha he aw eh ow o a e u".split())
# Any run of the same letter collapses to one before the base/hedge lookup. Real English almost
# never has triple letters, and the collapsed form is only matched against _FILLER_BASES / _HEDGES,
# so "book" -> "bok" is safely not-a-filler.
_FILLER_RE = re.compile(r"(.)\1+")


def _is_filler_token(token: str) -> bool:
    """Whether a single token is a filler / discourse marker. Stricter than a _FILLER lookup: it
    also catches drawn-out variants (via the collapse) and the hedges above."""
    clean = token.strip().strip(".,!?;:\"'()[]{}…-–—").casefold()
    if not clean:
        return True
    collapsed = _FILLER_RE.sub(r"\1", clean)
    return (clean in _FILLER or clean in _HEDGES
            or collapsed in _FILLER_BASES or collapsed in _HEDGES)


def _is_content_token(token: str) -> bool:
    """A token worth chasing as a concept: long enough, not a number, and not a filler."""
    clean = token.strip().strip(".,!?;:\"'()[]{}…-–—").casefold()
    return len(clean) > 2 and not clean.isnumeric() and not _is_filler_token(clean)


def _strip_filler(target: str) -> str:
    """Drop filler, sub-3-char, and numeric tokens from a (possibly multi-word) target span.
    Returns '' if nothing content remains — the signal callers use to decline recording / asking
    a question rather than ask one about a filler. Handles the multi-word spans the old single-
    token gate let through ("um basically" -> "")."""
    if not target:
        return ""
    return " ".join(t for t in target.split() if _is_content_token(t)).strip()


def is_concept_like(target: str) -> bool:
    """Whether a localized target is specific enough to chase.

    Filler firewall: a pause on "um" or "basically" is a delivery artefact, not a concept, so it
    must never enter the struggle ledger (which becomes the focus target, which becomes the next
    question). Single tokens AND multi-word spans are both stripped of filler; a span that is
    nothing but filler ("um basically") is rejected. Without this the student ends up asking
    "wait, I thought 'basically' was different?" — the canonical BS question.
    """
    return bool(_strip_filler(target))


# --- "I don't know" -------------------------------------------------------------------------
# Phrases only, deliberately NOT a length heuristic. "Friction." is a one-word answer that may be
# exactly right, and treating short answers as surrender would punish the concise learner — the
# opposite of what this is for. An answer with no content left after the filter counts too, since
# "um, yeah, like..." is a shrug however many words it took.
_GIVE_UP = (
    "i don't know", "i dont know", "i do not know", "dunno", "idk", "no idea", "not sure",
    "i'm not sure", "im not sure", "no clue", "i give up", "give up", "you tell me", "just tell me",
    "tell me the answer", "explain it to me", "i forgot", "i can't remember", "i cant remember",
    "i can't explain", "i cant explain", "skip", "pass", "next question",
)


def _is_give_up(answer: str) -> bool:
    """Whether an answer is the learner asking for help rather than attempting one.

    This is the branch that decides between asking again and teaching. Getting it wrong in the
    permissive direction hands over an answer the learner could have reached; getting it wrong in
    the strict direction strands them, which is the bug this exists to fix.
    """
    clean = answer.strip().casefold()
    if not _strip_filler(clean):
        return True
    return any(phrase in clean for phrase in _GIVE_UP)


def _mentions(text: str, concept: str) -> bool:
    """Whether an utterance mentions a ledger concept.

    Substring, not a word-set test: `localized_target` is frequently a multi-word judge span
    ("the Calvin cycle"), and comparing against set(text.split()) can never match one — the decay
    branch would look correct and silently never fire.
    """
    concept = concept.strip().casefold()
    return bool(concept) and concept in text.casefold()


def _decay_concepts(progress: ClassProgressRecord, *texts: str) -> None:
    """Halve the struggle score of every ledger concept named in `texts`, drop what falls below
    MASTERED_BELOW, and re-pick the focus target.

    Shared by the three ways a concept stops being outstanding: said clearly, answered correctly,
    or explained to them. Halving rather than deleting is what stops one lucky exchange erasing a
    real gap — and on the explained path it is also what stops the room teaching the same concept
    every single turn, since two explanations drop it off the top spot.
    """
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
    """Maintain the per-class struggle ledger and pick the concept to keep chasing.

    Accumulate, don't overwrite: a concept fumbled three times outranks one fumbled once, which is
    what makes the focus target stable enough to hold a conversation together. Mastery decays the
    score rather than deleting it outright, so one lucky clean sentence doesn't erase a real gap.
    """
    scores = progress.struggle_scores
    target = (analysis.localized_target or "").strip()
    if not is_concept_like(target):
        target = ""

    if target:
        struggle = sum(a.score for a in analysis.anomalies if a.type in STRUGGLE_ANOMALIES)
        if struggle > 0:
            scores[target] = round(scores.get(target, 0.0) + struggle, 4)

    # Proof of understanding: said clearly, nothing flagged. Credit every ledger concept it names.
    if analysis.confidence > MASTERY_CONFIDENCE and not analysis.anomalies:
        _decay_concepts(progress, analysis.text)

    if len(scores) > LEDGER_LIMIT:
        keep = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:LEDGER_LIMIT]
        progress.struggle_scores = dict(keep)
        scores = progress.struggle_scores

    progress.focus_target = max(scores, key=lambda k: scores[k]) if scores else ""


def _gpu_asked(analysis: ChunkAnalysis) -> bool:
    """Whether the ml-service itself judged this utterance worth a question.

    Its own gate, and a considered one — it fires on the Space B/C judges, which catch a wrong or
    self-contradicting claim delivered perfectly fluently, something a confidence threshold cannot
    see. Used as a trigger only; what actually gets asked is decided in `class_audio_turn`.

    Deliberately NOT keyed on `localized_target`, which is set for any anomaly at all. Firing on
    bare anomalies is what `settings.question_gate_on_anomalies` turns off by default, because the
    on-box judges are noisy on short utterances.
    """
    generated = analysis.student_question
    return (
        generated is not None
        and bool(generated.question_text.strip())
        and is_concept_like(generated.target_concept)
    )


def _gpu_hint(analysis: ChunkAnalysis) -> str:
    """The ml-service's guess at what broke, as a hint for our own question generator.

    Only a hint. It is one word chosen by acoustic score, so it says WHERE the utterance came apart
    and not what the learner was talking about — the generator is the one that decides what idea
    that word belonged to. Used only when our own struggle ledger is still empty, which is the
    first turn or two of a class.
    """
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
    """Relay the question the GPU wrote. Now a fallback only, for when our own generator is
    unreachable — see `class_audio_turn` for why it stopped being the primary author."""
    generated = analysis.student_question
    if generated is None or not generated.question_text.strip():
        return None
    # Written on the GPU from `target_concept`. If that was a filler word the question is about
    # nothing ("how does 'You' relate to CNNs?"); drop it and let the LLM generator, which sees the
    # transcript and the curriculum, ask something real instead.
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


# ---- concurrent writes to PathMemory ---------------------------------------------------------
# One writer at a time per path. PathMemory is a single JSONB blob with two writers that own
# different parts of it: the teaching turn owns the turn counter, the struggle ledger and
# asked_questions; `run_objective_check` owns the objective checkmarks. Whichever wrote last used
# to erase the other's work, and at one coverage check per chunk they overlap on nearly every turn
# — a checkmark would appear and then vanish, which reads as the goals lagging even further.
#
# The LLM calls stay OUTSIDE the lock. It is held only across re-read → merge → write.
_memory_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


async def _save_memory(
    path_id: str, class_id: str, cls: ClassUnit, memory: PathMemory
) -> None:
    """Persist a teaching turn's memory without dropping coverage that landed while it was busy.

    Only the fields `run_objective_check` owns are merged forward; everything else in `memory` is
    this turn's and wins.
    """
    async with _memory_locks[path_id]:
        stored = (await store.get_memory(path_id)).class_progress.get(class_id)
        progress = memory.class_progress.get(class_id)
        # The class was thrown away while this turn was talking to the model. Everything in
        # `memory` describes a lesson that no longer exists — write none of it.
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

    # Confusion gate over just-spoken utterance (mock heuristic — the SAME gate as
    # /questions/from_chunk, via engine.is_confused, so the two paths can't drift).
    analysis = engine.analyze([latest_utterance])[0]
    analysis.chunk_id = resp.new_segment.id
    if not engine.is_confused(analysis):
        await store.update_memory(path_id, memory)
        return ClassTeachResponse(
            student_reply=resp.student_reply, new_segment=resp.new_segment, asked=False
        )

    history = await store.get_history(session_id) + _memory_as_history(memory)
    # Same context as the audio path: a question written from one orphaned sentence has nothing to
    # anchor on but the words in it, which is how it ends up being about a word.
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
    """Throttle for goal nudges. Deliberately separate from `objective_check_every`: judging what
    was covered is the expensive batched call, whereas steering only needs the turn counter, and
    tying them together is why a probe used to need 3 chunks AND 4 turns and so never fired."""
    return progress.turn_count - progress.last_goal_probe_turn >= settings.goal_probe_cooldown


def _next_objective(
    open_objectives: list[ClassObjective], focus_target: str
) -> ClassObjective:
    """Which uncovered objective to nudge toward.

    Previously always `open_objectives[0]` — the first objective in CURRICULUM order, unrelated to
    anything the learner had just said. That is why the second question so often arrived as a
    non-sequitur: question one chased a stumble, and two turns later the probe jumped to objective
    o1. When the ledger knows what they keep tripping over, stay on it.
    """
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
    """A student asking about something the learner hasn't covered yet."""
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


# How many times a student presses on one concept before it stops asking and answers instead. A
# learner who cannot get there in three tries will not get there on the fourth, and by then the
# questions have stopped being a probe and started being a wall.
#
# Both caps count ANSWERS THE THREAD HAS TAKEN, including the one being graded right now — the
# answer is written to the ledger before the count is read. A cap of N means "the Nth answer is
# the one we stop asking on".
MAX_CONVERSATION_TURNS = 3

# The same cap for a question the verifier cannot grade. GPU-relayed questions carry no answer key
# and grading can only ever return "wrong" for them, so pressing three times is three rounds
# against a standard nobody can check — one press, then the answer. It must still be more than
# one: at one, the learner's very first reply ends the exchange, the student answers its own
# question, and the `?` disappears with no way to try again.
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
    """Grade one answer to a specific student's question and decide whether to press further.

    Returns (student_reply, follow_up, correct, conversation_over). The verifier is what makes this
    honest: previously any answer at all retired the question, so "uhh I dunno" and a real
    explanation were scored identically.

    The exchange only ever ends two ways now — they got there, or they were told. It used to end a
    third way, "LET'S COME BACK TO IT", which sounded like patience and was actually abandonment:
    the learner walked away from a concept they had just failed at, none the wiser.
    """
    question = answering.question
    turns = await store.thread_turns(session_id, question.id)
    objectives = [o.text for o in cls.checklist()]
    topic = f"{cls.title} — {cls.objective}"

    # A shrug is not an attempt, so don't spend a verifier call grading one against the key.
    gave_up = _is_give_up(answer)
    correct, _not_covered = (False, False) if gave_up else await grade_answer(answer, question)
    # No key to grade against (GPU-relayed questions carry none): grading always says wrong, so
    # this one gets a single press before we teach rather than three against a standard we can't
    # actually check.
    gradable = bool((question.answer_key or "").strip())
    cap = MAX_CONVERSATION_TURNS if gradable else UNGRADED_CONVERSATION_TURNS

    if correct:
        # Understood: let the ledger decay reflect it, and say so in character.
        _decay_concepts(progress, question.text, answer)
        return "OH — THAT MAKES SENSE NOW. THANKS!", None, True, True

    # They asked for help, or they have spent their tries. Answer the question.
    if gave_up or turns >= cap:
        said = await explain_answer(
            question,
            topic=topic,
            transcript=mastery.transcript_excerpt(transcript),
            objectives=objectives,
        )
        progress.explanations_given += 1
        # Being told counts as addressed, not as mastered: the score halves so the room stops
        # circling this concept, but a real gap survives one explanation and can come back.
        _decay_concepts(progress, question.text)
        return said, None, (None if not gradable else False), True

    # Still wrong and we have turns left: press on the SAME concept, from a new angle.
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
    """Use the ml-service transcript and confidence in one atomic teaching turn.

    `silent` is the live-classroom mode: the kid talks continuously and every pause ships a chunk,
    so the utterance is still transcribed, stored and analyzed — but the class only speaks when a
    question actually fires. Without it each chunk costs a reply LLM call and the room ends up
    several sentences behind the teacher. Set it False for the one-to-one flow, where the student
    is expected to answer whatever was just said to them.
    """
    if degraded or not analysis.text.strip():
        return AudioClassTeachResponse(analysis=analysis, degraded=True)

    session_id = class_session_id(path_id, class_id)
    # Everything said BEFORE this turn. Both stores now return a copy, so appending below does not
    # change it — the two spines are named separately here so each call site says which it wants.
    prior_transcript = await store.get_transcript(session_id)
    # Build the segment before deciding on a reply: the question generator below reads the stored
    # transcript, and it must see the same spine whether or not the student ends up speaking.
    segment = next_segment(prior_transcript, analysis.text.strip())
    analysis.chunk_id = segment.id
    await store.append_segment(session_id, segment)
    transcript_with_latest = prior_transcript + [segment]
    await store.append_analysis(session_id, analysis)

    memory = await store.get_memory(path_id)
    progress = _advance_progress(memory, class_id, cls)
    _apply_curriculum_update(memory, analysis)
    _update_struggle_ledger(progress, analysis)

    # Answering one student directly: grade it and either accept or press further. This path never
    # falls through to the classroom question logic below — a one-to-one conversation should not be
    # interrupted by an unrelated goal probe.
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

    # "I don't know how to explain this" said into the room is the same request for help as it is
    # face to face, and asking them another question is the one response guaranteed not to help.
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
            all_goals_covered=not open_objectives,
        )

    # WHETHER to ask is still both detectors: our own confusion gate, or the ml-service having
    # found an anomaly it thought was worth a question. Dropping the second one would make the
    # room deaf to everything Space B/C catches inside a confidently-delivered sentence.
    #
    # WHAT to ask is now decided here rather than on the GPU. The ml-service localizes confusion to
    # a single word by acoustic score, and a question templated onto that word is about a stumble
    # rather than a subject ("wait, I thought 'gradient' was different?"). This generator sees the
    # class goals, the recent spine and the concept the learner keeps tripping over, so it can name
    # the idea the fragment belonged to — and it writes an answer key, which is what lets the
    # conversation below grade an answer instead of accepting any noise as understanding.
    question: TargetedQuestion | None = None
    if engine.is_confused(analysis) or _gpu_asked(analysis):
        questions = await generate_targeted_questions(
            [analysis],
            await store.get_history(session_id) + _memory_as_history(memory),
            start_id=await store.next_question_id(session_id),
            topic=f"{cls.title} — {cls.objective}",
            transcript=excerpt,
            # The GPU's target is a usable hint when our own ledger has nothing yet — it says
            # WHERE the utterance broke, and the generator decides what that is about.
            focus_target=progress.focus_target or _gpu_hint(analysis),
            objectives=objectives,
        )
        if questions:
            question = questions[0]
        else:
            # Generator unreachable or unparseable: the GPU's relay is better than silence.
            question = await _question_from_analysis(session_id, analysis, memory)

    # Nothing wrong with what they said — but silence is not feedback. If goals are still open,
    # a student steers toward the next one; that turns dead air into direction, and is the only
    # signal the learner gets that the room is following at all.
    if question is None and open_objectives and _probe_is_due(progress):
        # WITH the latest utterance: the probe reacts to what was just said, and its chunk_id must
        # point at that segment rather than the one before it.
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
        student_reply = ""          # the class listened; nobody had anything to ask.
    else:
        # WITHOUT it: student_turn renders the latest utterance separately ("The kid just said:"),
        # so passing the post-append spine would put the same sentence in the prompt twice.
        student_reply = (await student_turn(prior_transcript, analysis.text.strip())).student_reply

    await _save_memory(path_id, class_id, cls, memory)
    return AudioClassTeachResponse(
        student_reply=student_reply,
        new_segment=segment,
        analysis=analysis,
        asked=question is not None,
        question=question,
        degraded=False,
        # No question and nothing left to cover: say so, so the UI can show that the class is
        # following rather than leaving the learner guessing whether anything is working.
        all_goals_covered=not open_objectives,
    )


async def end_class(
    path_id: str,
    class_id: str,
    cls: ClassUnit,
    completion_mode: str = "self-teaching",
) -> PathMemory:
    """'End class' action: fold the class into cross-class memory. The class title becomes a
    covered concept; answered probes count as understood, unanswered ones as still-struggled
    (so a later class may re-probe them from a new angle)."""
    session_id = class_session_id(path_id, class_id)
    memory = await store.get_memory(path_id)

    progress = memory.class_progress.get(class_id) or ClassProgressRecord()
    if progress.status == "complete":
        return memory
    now = time.time()
    progress.status = "complete"
    # Readiness stays as earned. Stamping 100 on every completed class was what made the whole
    # metric meaningless — "complete" says the learner stopped, `passed_on_mastery` says they got it.
    progress.readiness = _readiness(cls, progress)
    progress.passed_on_mastery = all(
        o.id in progress.covered_objectives for o in cls.checklist()
    )
    progress.started_at = progress.started_at or now
    progress.completed_at = now
    # Being told is not self-teaching, whatever the button that ended the class said. If a student
    # had to answer its own question even once, this class was guided.
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
        # An attempted answer wins: it's understood, and no longer counts as struggled. "I don't
        # know" is not an attempt — it is what earns an explanation, and a learner who was handed
        # the answer has not shown they can produce it. Crediting those was how a told concept got
        # skipped by every later class in the path.
        if entry.answer and not _is_give_up(entry.answer):
            if label not in memory.understood:
                memory.understood.append(label)
            if label in memory.struggled:
                memory.struggled.remove(label)
        elif label not in memory.struggled and label not in memory.understood:
            memory.struggled.append(label)

    # Objectives the learner demonstrably explained are understood across the whole path; the ones
    # left open are what a later class should come back to.
    for objective in cls.checklist():
        target = memory.understood if objective.id in progress.covered_objectives else memory.struggled
        if objective.text not in target:
            target.append(objective.text)

    await store.update_memory(path_id, memory)
    return memory


async def reset_class(path_id: str, class_id: str, cls: ClassUnit) -> PathMemory:
    """'Start this class over': erase everything this class recorded, and undo what it taught the
    rest of the path.

    For the lesson that went wrong under its own steam — a tangent that became the whole class,
    the wrong topic, ten minutes of thinking out loud. Ending such a class folds it into
    cross-class memory and later classes then quietly skip what it "covered"; abandoning it leaves
    the same debris. The learner needs a way to say "that wasn't it" and get the class they were
    given back.

    Scoped to this class. The session (speech, analyses, question ledger) goes entirely, and the
    concepts and questions it contributed to path memory are removed by name — the QA ledger is
    read before it is cleared precisely so they can be identified. `expanded_concepts` is left
    alone: those are beyond-scope concepts the learner genuinely raised, and nothing records which
    class raised them.
    """
    session_id = class_session_id(path_id, class_id)
    async with _memory_locks[path_id]:
        history = await store.get_history(session_id)
        # Everything path memory could be holding on this class's behalf: the questions it asked
        # and the goals it was judged against.
        own = {e.question.text for e in history} | {o.text for o in cls.checklist()}
        memory = await store.get_memory(path_id)
        previous = memory.class_progress.get(class_id)
        # A fresh record rather than a deleted key: `reset_count` is what an in-flight teaching
        # turn compares itself against before writing (see _save_memory), and a missing record
        # looks exactly like a class nobody has taught yet.
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


# ---- objective mastery (background) ----

# How much speech the judge sees. Wider than the unjudged tail on purpose: an objective explained
# across a batch boundary — half before the last check, half after — is invisible to a judge shown
# only the new half, and would sit open until the learner happened to repeat themselves.
COVERAGE_WINDOW_SEGMENTS = 6


async def run_objective_check(path_id: str, class_id: str, cls: ClassUnit) -> None:
    """Judge which objectives the learner has now covered.

    Runs as a FastAPI background task after the teaching response has already been sent, so the
    teaching loop never waits on it. One verifier call, and only when there is new speech to judge.

    Failures are swallowed: a missed check just means the checkmarks land a turn later, which is
    not worth breaking a live class over.
    """
    try:
        session_id = class_session_id(path_id, class_id)
        transcript = await store.get_transcript(session_id)
        if not transcript:
            return

        memory = await store.get_memory(path_id)
        progress = memory.class_progress.get(class_id) or ClassProgressRecord()
        judged_reset_count = progress.reset_count
        # `fresh` is only the gate — "has anything been said since the last check". What the judge
        # actually reads is the window below.
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

        # Re-read under the lock and merge onto whatever landed while the judge was thinking,
        # rather than writing back the snapshot this task started from.
        async with _memory_locks[path_id]:
            memory = await store.get_memory(path_id)
            progress = memory.class_progress.get(class_id) or ClassProgressRecord()
            # The learner started the class over while the judge was reading. The speech these
            # checkmarks were earned on has been deleted; awarding them now would tick goals off
            # a class that has not been taught yet.
            if progress.reset_count != judged_reset_count:
                return
            for objective_id, evidence in covered.items():
                if objective_id not in progress.covered_objectives:
                    progress.covered_objectives.append(objective_id)
                progress.objective_evidence[objective_id] = evidence
            progress.last_checked_segment = max(progress.last_checked_segment, latest_segment)
            progress.readiness = _readiness(cls, progress)
            # Steering lives on the teaching turn itself (see class_audio_turn) so the room reacts
            # while the learner is still there, not a turn later.
            memory.class_progress[class_id] = progress
            await store.update_memory(path_id, memory)
    except Exception as exc:  # noqa: BLE001 — a background nicety must never kill a live class
        print(f"[mastery] objective check failed for {path_id}:{class_id} ({exc!r})")
