# Learn by Teaching — Plan API

The "learn by teaching" flow: a learner names a topic, we build a **plan of ~5 classes**, generate
a brief **teacher's-notes** primer before each class, then the learner **teaches the class to an AI
student**. The AI knows nothing and only asks / restates / admits confusion — and when the learner
*sounds unsure*, it fires one targeted question about exactly that spot. **Cross-class memory**
keeps it from asking near-duplicate questions across classes.

This layer is additive: it reuses the existing confusion gate + targeted-question pipeline. All
endpoints are under the `/plan` prefix and are auto-documented at `/docs`.

> Every example below is **real captured output** from the running app (DeepSeek `deepseek-chat`).

---

## The flow at a glance

```
 "I want to learn physics"
        │
        ▼
 ①  POST /plan/scope ───────────► too broad? → 3 narrower options
        │                          scoped?    → confirmed topic
        │  (learner picks a topic)
        ▼
 ②  POST /plan/build ───────────► GrowthPath: ~5 ordered classes,
        │                          titles + objectives + every class's
        │                          teacher's notes, written together so
        │                          no two classes teach the same thing
        │
        │  ┌──────────────── for each class, in order ───────────────┐
        ▼  ▼                                                          │
 ③  POST /plan/{id}/class/{cid}/notes  ─► the primer, rewritten       │
        │      (backfill / ?regenerate=true — normally already there) │
        ▼                                                             │
 ④  POST /plan/{id}/class/{cid}/teach/turn  (repeat many times)       │
        │   learner teaches → AI student replies;                     │
        │   a question fires ONLY when the learner sounds unsure       │
        ▼                                                             │
 ⑤  POST /plan/{id}/class/{cid}/end   ("End class")                   │
        │   folds the class into cross-class memory                   │
        └──────────────── then the next class → ③ ───────────────────┘
```

Each class runs on its own store session: `session_id = "{path_id}:{class_id}"`.

---

## Data models

```python
GrowthPath:                          # the learning plan
  path_id: str                       # "gp-18d8e54a" (server-assigned)
  original_input: str                # "I want to learn physics"
  confirmed_topic: str               # "Classical Mechanics: Forces and Motion"
  total_classes: int
  recommended_order: list[str]       # ["c1","c2","c3"] — teaching sequence
  classes: list[ClassUnit]
  source_material_summary: str|None  # short summary if material_text was pasted

ClassUnit:                           # one class = one topic (no subtopics)
  class_id: str                      # "c1"
  title: str                         # "Introduction to Forces"
  objective: str                     # one-sentence learning goal
  difficulty: str                    # "beginner" | "intermediate" | "advanced"
  prerequisites: list[str]           # class_ids that should come first
  teacher_notes: str                 # Markdown primer (written by /plan/build)
  notes_generated: bool              # false only if that write failed, or on an older path

PathMemory:                          # cross-class memory (per path_id)
  path_id: str
  covered_concepts: list[str]        # already taught → notes don't re-teach
  asked_questions: list[str]         # already asked → no near-duplicates
  understood: list[str]              # learner answered the probe
  struggled: list[str]               # learner left the probe unanswered
  expanded_concepts: list[str]       # valid beyond-scope concepts found during audio teaching
```

---

## ① `POST /plan/scope` — confirm or narrow the topic

Decide whether the request is teachable as-is. If it spans multiple domains (e.g. "physics"), it's
**too broad** and we return 3 narrower options for the learner to choose from.

**Request** — `ScopeRequest`
```json
{
  "original_input": "I want to learn physics",
  "material_text": null,
  "preferred_classes": null
}
```

**Response** — `TopicScope` (this input is too broad)
```json
{
  "is_broad": true,
  "suggestions": [
    {
      "topic": "Classical Mechanics: Forces and Motion",
      "rationale": "Covers Newton's laws, kinematics, and dynamics — the foundational pillars of physics that are self-contained and teachable in a focused sequence.",
      "suggested_classes": 6
    },
    {
      "topic": "Electricity and Magnetism: Circuits and Fields",
      "rationale": "A coherent domain covering charge, current, voltage, resistance, and basic circuit analysis without needing other physics subfields.",
      "suggested_classes": 6
    },
    {
      "topic": "Waves and Sound: Properties and Behavior",
      "rationale": "Focuses on wave types, superposition, frequency, amplitude, and sound phenomena as a standalone topic with clear learning objectives.",
      "suggested_classes": 5
    }
  ],
  "confirmed_topic": "Classical Mechanics: Forces and Motion",
  "suggested_classes": 6
}
```

- **Frontend:** if `is_broad`, show the `suggestions` as pickable cards. If `is_broad: false`,
  `suggestions` is empty and `confirmed_topic` is ready to build directly.

---

## ② `POST /plan/build` — build the plan, material and all

Turns a confirmed topic into an ordered set of classes **and writes every class's teacher's notes**.

Two model steps: one structures the course (titles, objectives, order), then one call per class —
fired in parallel, capped at `settings.notes_concurrency` — writes the material. Each of those
calls is shown the whole outline, split into the classes taught *before* this one and the ones
taught *after*, with their objectives. That is what stops class 3 re-teaching class 1: a primer
written alone knows only its own title and has to guess where its neighbours' ground begins.

So this is the slow call in the plan surface (N+1 model calls, tens of seconds). If a class's
notes call fails it comes back with `notes_generated: false` and ④ fills it on demand — a
rate-limited notes tier never costs the learner the course.

**Request** — `BuildPlanRequest`
```json
{
  "original_input": "I want to learn physics",
  "confirmed_topic": "Classical Mechanics: Forces and Motion",
  "num_classes": 3,
  "material_text": null
}
```

**Response** — `GrowthPath`
```json
{
  "path_id": "gp-18d8e54a",
  "original_input": "I want to learn physics",
  "confirmed_topic": "Classical Mechanics: Forces and Motion",
  "total_classes": 3,
  "recommended_order": ["c1", "c2", "c3"],
  "classes": [
    {
      "class_id": "c1",
      "title": "Introduction to Forces",
      "objective": "Learner will be able to define a force, identify common types of forces, and explain how forces affect the motion of an object.",
      "difficulty": "beginner",
      "prerequisites": [],
      "teacher_notes": "## Introduction to Forces\n\nA **force** is simply a push or a pull...",
      "notes_generated": true
    },
    {
      "class_id": "c2",
      "title": "Newton's Laws of Motion",
      "objective": "Learner will be able to state and apply Newton's three laws of motion to predict the behavior of objects under various forces.",
      "difficulty": "beginner",
      "prerequisites": ["c1"],
      "teacher_notes": "## Newton's Laws of Motion\n\nYou already know what a force does...",
      "notes_generated": true
    },
    {
      "class_id": "c3",
      "title": "Applications of Forces and Motion",
      "objective": "Learner will be able to solve real-world problems involving friction, tension, and inclined planes using free-body diagrams and Newton's laws.",
      "difficulty": "intermediate",
      "prerequisites": ["c1", "c2"],
      "teacher_notes": "## Applications of Forces and Motion\n\nFree-body diagrams turn a word...",
      "notes_generated": true
    }
  ],
  "source_material_summary": null
}
```

- `num_classes` is optional (defaults to `settings.default_classes` = 5). An explicit `0` is not
  silently swapped for the default; it's clamped to a minimum of 1.
- `material_text` is used in full here (the first 4 000 chars per class) — this is the only moment
  the whole upload is in hand, since only a 500-char `source_material_summary` is stored.
- The plan is **saved** — fetch it again anytime with ③.

---

## ③ `GET /plan/{path_id}` — fetch the plan

Returns the stored `GrowthPath` (same shape as ②), `teacher_notes` and all. `404` if the `path_id`
is unknown.

```
GET /plan/gp-18d8e54a  →  200  { ...GrowthPath... }
```

---

## ④ `POST /plan/{path_id}/class/{class_id}/notes` — rewrite one class's notes

The **brief Markdown primer** (~200–400 words) the learner reads to prepare to teach. May embed
**one** ```` ```mermaid ```` diagram; **never images**.

② normally wrote this already, so the route is the backfill and exists for three cases: a path
built before notes were eager, a class whose eager write failed, and `?regenerate=true` once the
learner has actually taught some classes — that last one is the only version of the primer that
can see cross-class **memory**, which is empty at build time by definition.

Already-generated notes are returned as-is; pass `?regenerate=true` to spend the call.

**Request:** none (path params only).

**Response** — the enriched `ClassUnit` (`notes_generated` flips to `true`)
```json
{
  "class_id": "c1",
  "title": "Introduction to Forces",
  "objective": "Learner will be able to define a force, identify common types of forces, and explain how forces affect the motion of an object.",
  "difficulty": "beginner",
  "prerequisites": [],
  "teacher_notes": "## Introduction to Forces\n\nA **force** is simply a push or a pull...\n\n## Key Ideas\n...\n\n## Common Pitfalls\n...",
  "notes_generated": true
}
```

The `teacher_notes` string is Markdown. Rendered, it looks like:

> ## Introduction to Forces
>
> A **force** is simply a push or a pull. It's the way objects interact with each other. Forces are
> the "engines" of motion... We measure force in **newtons (N)**. A force has both a **size**
> (magnitude) and a **direction**, making it a **vector** quantity.
>
> ## Key Ideas
> ### How Forces Affect Motion
> - A force can **start** an object moving, **stop** it, **speed it up**, **slow it down**, or **change its direction**.
> - If multiple forces act on an object, we consider the **net force**.
> - **Balanced forces** (net = 0) → no change. **Unbalanced forces** (net ≠ 0) → acceleration.
>
> ### Common Types of Forces
> - **Gravity**, **Friction**, **Normal force**, **Tension**, **Applied force** …
>
> ## Common Pitfalls
> - **Forces don't need contact** — gravity works at a distance.
> - **Motion doesn't require a continuous force** … (covered in Newton's First Law)
> - **Don't confuse mass and weight** …

> **Note:** for a class that has a structural/flow relationship, the notes may include a fenced
> ` ```mermaid ` block — the frontend renders it as a diagram (no image files involved).

---

## ⑤ `POST /plan/{path_id}/class/{class_id}/teach/turn` — teach a turn

The learner says something; the AI student replies in character. The utterance runs through the
**confusion gate** (`engine.is_confused` — the *same* gate as the real-time endpoint). A question
fires **only** when the learner sounds unsure, and never repeats a question already asked anywhere
in this path.

**Request** — `TeachTurnBody`
```json
{ "latest_utterance": "A force is a push or a pull that changes an object's motion." }
```

**Response — confident utterance (`asked: false`)**
```json
{
  "student_reply": "Wait... so if I push my pencil across the desk, that's a force, but what about when I just hold it still in my hand? Is that still a force?",
  "new_segment": { "id": 0, "idx": 0, "text": "A force is a push or a pull that changes an object's motion.", "t_start": null, "t_end": null },
  "asked": false,
  "question": null
}
```

**Request — a hesitant utterance**
```json
{ "latest_utterance": "um, i think a force is maybe like, kind of a push? i'm not sure." }
```

**Response — unsure utterance (`asked: true`, a targeted question fires)**
```json
{
  "student_reply": "Wait, so is it just a push, or can it also be a pull? I thought you said it changes motion.",
  "new_segment": { "id": 1, "idx": 1, "text": "um, i think a force is maybe like, kind of a push? i'm not sure.", "t_start": null, "t_end": null },
  "asked": true,
  "question": {
    "id": 0,
    "chunk_id": 1,
    "text": "You said a force is 'kind of a push' — what else could a force be, if not just a push?",
    "anomaly_type": "hedging",
    "rationale": "The learner hedged with 'kind of a push' and 'i'm not sure.' This question draws them out on the other half of the definition (pull) without revealing the answer."
  }
}
```

- The gate fires on low confidence **or** lexical hesitation markers ("um", "maybe", "kind of",
  trailing "?"). A clear statement produces `asked: false`.
- To record the learner's answer to a fired question (so it counts as *understood* at end-of-class),
  reuse the existing `POST /questions/answer` with `session_id = "{path_id}:{class_id}"` and the
  `question.id`.

---

## `POST /plan/{path_id}/class/{class_id}/teach/audio-turn` — context-aware audio turn

Send one recorded utterance as multipart form-data (`audio`, optional `chunk_id`, `history` and
`silent`). The route automatically supplies the GPU service with the current path topic, class
objective, teacher notes, source-material summary, covered concepts, and previously expanded
concepts.

The response contains the saved transcript segment and full `ChunkAnalysis`. When the GPU returns
`student_question`, that question becomes the AI Student reply and is added to cross-class question
memory. When it returns `curriculum_update.added_concepts`, those concepts are persisted in
`PathMemory.expanded_concepts` and included in later turns. If the GPU service is unavailable, the
response explicitly sets `degraded: true` and does not fabricate a transcript or question.

### `silent` — the live classroom

`silent=true` is how the frontend teaches. The learner presses the mic, teaches, and presses again
to ship the chunk (the mic is never live on its own), so the utterance is still transcribed, stored
as a segment and analyzed — but the class only speaks when a question actually fires:

| | question fired | no question |
|---|---|---|
| `silent=false` (default) | `student_reply` = the question | `student_reply` = an LLM reply |
| `silent=true` | `student_reply` = the question | `student_reply = ""`, **no LLM call** |

Without it every chunk costs a reply round-trip and the class ends up several sentences behind the
teacher. The transcript is identical either way, so the end-of-class `/analysis/{session_id}`
measurement is unaffected — which is exactly why the segment is recorded even when nobody speaks.

Use `silent=false` for a one-to-one exchange, where a student asked something and is expected to
answer back.

### `explained` — "I don't know"

The one thing that speaks even when `silent=true`. A learner who says they are stuck — "I don't
know", "I'm not sure", "no idea", or nothing but filler — is asking for the answer, and another
question is the single response guaranteed not to help. That turn returns the explanation as
`student_reply` with **`explained: true`**, no question attached, and `explanations_given`
incremented. Both the audio and text turns do this, and the flag is what lets a client tell an
answer it must show from a line of student chatter it can drop.

Detection is by phrase *and position*: the admission has to fall within the first few words. People
front-load surrender and bury a hedge mid-sentence, so "honestly, I don't know" hands over the
answer while "energy is sort of, I dunno, the ability to do stuff" is treated as the attempt it is
and earns a question instead. Being told halves the concept's struggle score rather than clearing
it — a gap survives one explanation and can come back.

---

## ⑥ `POST /plan/{path_id}/class/{class_id}/end` — "End class"

Folds the class into cross-class memory: the class title becomes a covered concept; answered probes
count as *understood*, unanswered probes as *struggled*.

**Request:** none (path params only).

**Response** — `PathMemory`
```json
{
  "path_id": "gp-18d8e54a",
  "covered_concepts": ["Introduction to Forces"],
  "asked_questions": [
    "You said a force is 'kind of a push' — what else could a force be, if not just a push?"
  ],
  "understood": [],
  "expanded_concepts": ["environmental decoherence"],
  "struggled": [
    "You said a force is 'kind of a push' — what else could a force be, if not just a push?"
  ]
}
```

---

## `POST /plan/{path_id}/class/{class_id}/reset` — "Start this class over"

The inverse of ⑥, for the lesson that went off the rails: a tangent that ate the class, the wrong
topic, ten minutes of thinking out loud. Without it the only exits were finishing a class the
learner knows is bad — which grades it and folds it into memory — or abandoning it half-taught.

Erases the class's store session (transcript, analyses, question ledger, any analysis job) and
replaces its `ClassProgressRecord` with a fresh one, then takes back what the class contributed to
cross-class memory: its title leaves `covered_concepts`, and the questions it asked (read off the
ledger before it is cleared) plus its objective texts leave `asked_questions`, `understood` and
`struggled`. `expanded_concepts` stays — those are beyond-scope concepts the learner genuinely
raised and nothing records which class raised them.

The plan and the teacher's notes are untouched: the learner gets the same class back, not a new one.

`ClassProgressRecord.reset_count` increments on every reset. It is not a statistic — a teaching turn
or background coverage check that was mid-flight when the class was thrown away compares it against
the value it read and drops its write, so a deleted lesson cannot write itself back.

**Request:** none (path params only). **Response:** the updated `PathMemory`.

---

## How the cross-class memory works

`PathMemory` (one record per `path_id`) is the durable "what has happened on this path so far":

- **`asked_questions`** — on every teaching turn, all previously-asked questions (across *every*
  class) are handed to the question generator as "do **NOT** repeat any of these." Because the
  generator is also told to "ask something genuinely new," a learner who keeps hesitating on the
  same idea still gets re-probed — from a fresh angle. That's the "only re-ask if they don't get it"
  behavior.
- **`covered_concepts`** — feeds the notes generator ("build on these; don't re-teach"). Also seeded
  from the titles of classes earlier in `recommended_order`, so notes stay coherent even if you
  generate them before teaching the earlier classes.
- **`understood` / `struggled`** — a lightweight per-class signal from which probes got answered.
- **`expanded_concepts`** — valid above-and-beyond concepts returned by the live GPU analysis;
  they become context for subsequent class turns instead of being discarded.

---

## Running it

```bash
pip install -r requirements.txt
uvicorn app.main:app --port 8000          # in-memory store (dev default)
# open http://localhost:8000/docs for the interactive schema
```

Config knobs (`app/config.py` / `.env`): `GENERATOR_MODEL` / `GENERATOR_BASE_URL` (DeepSeek by
default), `default_classes` (5), and the confusion gate's `question_confidence_threshold` (0.5).
For durability across restarts run with `STORE_BACKEND=db` + `DATABASE_URL` (Postgres) — the plan
and memory persist to the `growth_paths` / `path_memory` tables.

### Minimal end-to-end (curl)

```bash
BASE=http://localhost:8000

# ① scope
curl -s $BASE/plan/scope -H 'content-type: application/json' \
  -d '{"original_input":"I want to learn physics"}'

# ② build (pick a confirmed topic from the scope response)
PID=$(curl -s $BASE/plan/build -H 'content-type: application/json' \
  -d '{"original_input":"I want to learn physics","confirmed_topic":"Classical Mechanics: Forces and Motion","num_classes":3}' \
  | python -c 'import sys,json;print(json.load(sys.stdin)["path_id"])')

# ③ the first class's notes — already written by the build; this just reads them back
curl -s $BASE/plan/$PID/class/c1/notes

# ④ teach a turn
curl -s $BASE/plan/$PID/class/c1/teach/turn -H 'content-type: application/json' \
  -d '{"latest_utterance":"um, i think a force is maybe kind of a push?"}'

# ⑤ end the class
curl -s $BASE/plan/$PID/class/c1/end
```
