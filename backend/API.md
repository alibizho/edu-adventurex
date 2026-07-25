# Teachable Student — Backend API & Frontend Guide

Everything a frontend developer needs to talk to the backend: how to run it, the two UX flows,
every endpoint with exact params + response shapes, and the gotchas.

## Run it

```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --port 8000     # or --reload for dev
```

- **Base URL:** `http://localhost:8000`
- **Interactive docs:** `http://localhost:8000/docs` (Swagger — try endpoints live) and `/redoc`.
- **CORS:** open (`*`) — any frontend origin/port can call it directly, no proxy needed.
- **State:** in-memory by default (lost on restart). Run with `STORE_BACKEND=db` + `DATABASE_URL`
  (or `docker compose up`) for durable Postgres storage that survives restarts. `session_id` is any
  client-chosen string (a UUID is fine) — one per teaching session.

## The two UX flows

There are two flows. **Flow A is the primary spoken-class experience.** Flow B is the text-based
teaching loop plus the measurement/analytics endpoints. They share a `session_id` but don't yet
share all state (see the gap note below).

### Flow A — Real-time spoken class (primary)

The human states a topic, then teaches a "class" of AI students by speaking. The frontend records
audio, detects pauses (a few seconds of silence via VAD), and sends each chunk to the backend. The
backend runs the chunk through the confusion engine (on the GPU box) and — **only if the chunk
sounded confused** — generates one targeted question about it. The AI students stay silent; the
question is the output.

1. Collect a **topic** string from the user before they start (e.g. `"how packet routing works"`).
2. Record audio; split it into chunks at pauses (client-side VAD — the backend does not stream).
3. For each chunk, `POST /questions/from_chunk` with the audio file, `session_id`, an incrementing
   `chunk_id`, and `topic` (send it at least on the first call — it's stored on the session and
   reused after that).
4. Render the response:
   - `asked: true` → show `question.text` (frame it as a student raising a hand, or a prompt for the
     teacher to clarify that exact spot). `question.rationale` explains why it was asked (debug/UI).
   - `asked: false` → no question. You can still show `analysis.text` (live transcript) and
     `analysis.confidence` subtly, or show nothing.
5. When the teacher answers a question, `POST /questions/answer` so the agent won't re-ask it.

The "confused" gate is: low confidence (`analysis.confidence < 0.5`) **or** a hesitation marker
("um", "uh", a trailing "?"). The ml-service's anomaly flags are currently **not** used to trigger
questions (they're noisy on short utterances) — they're still returned in `analysis.anomalies` for
display. See **Practical notes**.

### Flow B — Text teaching + measurement (analytics)

A text-based teaching loop that builds a transcript, then measures how much understanding
transferred and fuses it with the confusion signal.

1. `POST /teach/turn` repeatedly — teacher's utterance in, student's in-character reply + a new
   transcript segment out. This is what builds the lesson transcript.
2. Analyze chunks for confusion: `POST /confusion/analyze` (audio → GPU) or `POST /confusion/mock`
   (text heuristic, no GPU needed).
3. `POST /measure?session_id=...` — runs the taught-vs-cold student ensemble and returns the
   transfer delta. **Slow (~1–2 min)** — show a loading state.
4. `GET /fusion/{session_id}` — crosses confusion (disturbance) × transfer delta → per-segment
   quadrants + a calibration score.
5. `POST /questions/next` + `POST /questions/answer` — targeted Q&A on the lowest-confidence chunks.

> **Known gap (important):** Flow A (spoken) stores confusion **analyses** but does **not** yet
> append to the teaching **transcript** that `/measure` and `/fusion` read. So a pure spoken session
> will get `404` from `/measure` ("no transcript for session …"). To get analytics on a spoken
> session today, also drive `/teach/turn` with the transcribed text (`analysis.text` from
> `/questions/from_chunk`). A future change will auto-append spoken chunks as transcript segments.

## Endpoint reference

| Method | Path | Request | Returns | Purpose |
|---|---|---|---|---|
| GET | `/health` | — | `{"ok": true}` | Backend liveness. |
| GET | `/` | — | `{"service","docs"}` | Service banner. |
| GET | `/confusion/health` | — | `{reachable, ok, device, …}` | Is the GPU ml-service up? **Call before audio flows.** |
| POST | `/teach/turn` | JSON `TeachTurnRequest` | `TeachTurnResponse` | One teaching turn → student reply + new segment. |
| POST | `/questions/from_chunk` | multipart form | `ChunkQuestionResponse` | **Flow A:** audio chunk → (maybe) a question. |
| POST | `/confusion/analyze` | multipart form | `ChunkAnalysis` | Audio chunk → analysis, optional GPU question, and curriculum update. |
| POST | `/plan/{path_id}/class/{class_id}/teach/audio-turn` | multipart form | `AudioClassTeachResponse` | Context-aware audio teaching turn used by the frontend. `silent=true` records + analyzes the chunk but only speaks to ask (see LEARN_BY_TEACHING.md). |
| POST | `/plan/{path_id}/class/{class_id}/reset` | — | `PathMemory` | "Start this class over": erases the class's session (speech, analyses, question ledger) and its progress, and takes back the concepts and questions it contributed to cross-class memory. The plan and teacher's notes are untouched. |
| POST | `/confusion/ingest` | JSON `IngestRequest` | `{session_id, n_chunks}` | Bulk-store precomputed analyses. |
| POST | `/confusion/mock` | `?session_id=` | `{session_id, n_chunks}` | Text-only heuristic analyses (no GPU). |
| POST | `/measure` | `?session_id=` | `RunResult` | Transfer-delta measurement (slow). |
| GET | `/fusion/{session_id}` | — | `FusionResult` | Disturbance × delta quadrant map. |
| POST | `/questions/next` | JSON `NextQuestionsRequest` | `TargetedQuestion[]` | Generate questions for the weakest chunks. |
| POST | `/questions/answer` | JSON `AnswerRequest` | `{session_id, question_id, recorded}` | Record an answer (prevents re-asking). |
| GET | `/questions/history/{session_id}` | — | `QAEntry[]` | The Q&A ledger for a session. |

---

### `POST /questions/from_chunk`  — the real-time flow

**Multipart form-data** (not JSON):

| Field | Type | Required | Notes |
|---|---|---|---|
| `session_id` | string | yes | Client-chosen session id. |
| `chunk_id` | int | no (default `0`) | Increment per chunk. Used as the chunk's identity. |
| `audio` | file | yes | WAV, ideally 16 kHz mono. |
| `topic` | string | no | The lesson topic. Stored on the session; send once, reused after. |
| `history` | string (JSON array) | no | Prior transcript texts for context. If omitted, the session's prior chunk transcripts are used. |
| `enable_space_c` | bool | no (default `false`) | Fact-check on/off. Enable when curriculum context is supplied. |
| `curriculum_context` | string | no | Current objective, notes, and source context for Space C. |
| `key_concepts` | string (JSON array) | no | Known and newly expanded concepts. |

**Response — `ChunkQuestionResponse`:**
```json
{
  "asked": true,
  "analysis": {
    "chunk_id": 0,
    "text": "Um, so the packet goes to, the router?",
    "confidence": 1.0,
    "localized_target": null,
    "anomalies": [],
    "detail": []
  },
  "question": {
    "id": 0,
    "chunk_id": 0,
    "text": "Can you describe in more detail what the router does when it receives a packet?",
    "anomaly_type": null,
    "rationale": null
  }
}
```
When `asked` is `false`, `question` is `null` (but `analysis` is still present).

**Frontend example (JS):**
```js
const form = new FormData();
form.append("session_id", sessionId);
form.append("chunk_id", String(chunkId));
form.append("topic", topic);                         // at least on the first chunk
form.append("audio", audioBlob, "chunk.wav");        // Blob from MediaRecorder

const res = await fetch("http://localhost:8000/questions/from_chunk", {
  method: "POST",
  body: form,                                        // don't set Content-Type; browser sets it
});
const { asked, analysis, question } = await res.json();
if (asked) showQuestion(question.text);
showTranscript(analysis.text, analysis.confidence);
```

### `POST /teach/turn` — one teaching turn

**JSON body — `TeachTurnRequest`:**
```json
{
  "session_id": "abc123",
  "transcript": [ {"id": 0, "idx": 0, "text": "..."} ],   // prior segments; [] on first turn
  "latest_utterance": "A message is split into packets."
}
```
**Response — `TeachTurnResponse`:**
```json
{
  "student_reply": "So each packet takes its own path to get to the same place?",
  "new_segment": {"id": 1, "idx": 1, "text": "A message is split into packets.", "t_start": null, "t_end": null}
}
```
Append `new_segment` to the transcript you send on the next turn.

### `POST /measure?session_id=abc123` — transfer delta

No body. **Slow (~1–2 min)** — the backend fans out an ensemble of AI students. Show a loading
state; consider running it in the background and polling, or just blocking with a spinner.

**Response — `RunResult`:**
```json
{
  "session_id": "abc123",
  "delta_overall": 0.33,          // taught_mean - cold_mean; >0 means teaching helped
  "survival_rate": 0.25,          // fraction of candidate questions that survived the filter
  "per_question": [
    {"question_id": 7, "taught_mean": 0.67, "cold_mean": 0.0, "delta": 0.67}
  ],
  "calibration_rho": null
}
```

### `GET /fusion/{session_id}` — quadrant map

Crosses the stored confusion analyses (disturbance = 1 − confidence) with the latest `/measure` run
(transfer delta). Call `/confusion/analyze` (or `/mock`) **and** `/measure` first; either alone
returns partial results (uncrossed segments show `quadrant: "unknown"`).

**Response — `FusionResult`:**
```json
{
  "session_id": "abc123",
  "per_segment": [
    {"segment_id": 0, "text": "...", "disturbance": 0.3, "transfer_delta": -0.29, "quadrant": "blind_spot"}
  ],
  "quadrant_counts": {"blind_spot": 3, "mastery": 0, "aware_gap": 0, "productive_struggle": 0, "unknown": 0},
  "calibration_rho": -1.0
}
```

**Quadrants** (disturbance × transfer delta):
- `blind_spot` — sounded clear but didn't learn (low disturbance, negative delta).
- `aware_gap` — sounded confused and didn't learn (high disturbance, negative delta).
- `productive_struggle` — sounded confused but learned (high disturbance, positive delta).
- `mastery` — sounded clear and learned (low disturbance, positive delta).
- `unknown` — not enough data to classify.

### `POST /confusion/analyze` — audio → analysis only

Same multipart shape as `/questions/from_chunk` (`session_id`, `chunk_id`, `audio`, optional
`history`, `enable_space_c`, `overall_topic`, `curriculum_context`, and `key_concepts`). It returns
the complete `ChunkAnalysis`, including an optional GPU-generated `student_question` and optional
`curriculum_update`. Use this when you want the signal without running the main-backend fallback
question generator, or to feed `/fusion`.

### `POST /questions/next` — questions for the weakest chunks

**JSON body — `NextQuestionsRequest`:** `{"session_id": "abc123", "n": 3}`
**Response:** `TargetedQuestion[]` — picks the `n` lowest-confidence chunks that don't already have
an answered question, and writes one question each. Requires analyses to already be stored (via
`/confusion/analyze`, `/confusion/mock`, `/confusion/ingest`, or `/questions/from_chunk`).

### `POST /questions/answer` — record an answer

**JSON body — `AnswerRequest`:** `{"session_id": "abc123", "question_id": 0, "answer": "..."}`
Returns `{"session_id", "question_id", "recorded": true}`. Marks the question answered so
`/questions/next` won't re-ask the same chunk.

### Smaller endpoints

- `GET /confusion/health` → `{"reachable": true, "ok": true, "device": "cuda", "whisper": "...", "space_c": true, "judge": "api", "vram_gb": 1.2}` (or `{"reachable": false, "error": "..."}`). **Call this before relying on audio flows** — if the GPU box is down, audio endpoints degrade to a neutral analysis.
- `POST /confusion/ingest` — body `{"session_id", "chunks": [ChunkAnalysis, …]}` → `{session_id, n_chunks}`. Bulk-load analyses computed elsewhere.
- `POST /confusion/mock?session_id=abc123` → `{session_id, n_chunks}`. Runs the text heuristic over the stored transcript; no GPU needed. Good for offline dev.
- `GET /questions/history/{session_id}` → `QAEntry[]` where each entry is `{"question": TargetedQuestion, "answer": "..." | null, "answered_at": 1234.5 | null}`.

## Key data shapes

```
Segment          { id, idx, text, t_start?, t_end? }
ChunkAnalysis    { chunk_id, text, confidence, anomalies: [Anomaly], localized_target?, detail: [WordScore], student_question?, curriculum_update? }
StudentQuestion  { question_text, target_concept, anomaly_type }
CurriculumUpdate { added_concepts: [string] }
Anomaly          { type, source, score, evidence? }     # type: factual_error | logic_error | recall_failure | hedging
TargetedQuestion { id, chunk_id, text, anomaly_type?, rationale? }
ChunkQuestionResponse { asked: bool, analysis: ChunkAnalysis, question: TargetedQuestion? }
RunResult        { session_id, delta_overall, survival_rate, per_question: [QuestionDelta], calibration_rho? }
QuestionDelta    { question_id, taught_mean, cold_mean, delta }
FusionResult     { session_id, per_segment: [SegmentFusion], quadrant_counts, calibration_rho? }
SegmentFusion    { segment_id, text, disturbance, transfer_delta?, quadrant }
```

## Practical notes

- **`confidence` semantics:** `∈ [0,1]`, **HIGH = speaker sounded clear**, LOW = confused.
  `disturbance = 1 − confidence`. Don't invert it in the UI.
- **Audio format:** WAV, 16 kHz mono is ideal (matches Whisper). Record with `MediaRecorder`, send
  the Blob. The backend forwards it to the GPU ml-service, so keep chunks short (a few seconds — one
  spoken utterance) for fast turnaround.
- **`/questions/from_chunk` latency:** one GPU round-trip (Whisper + confusion engine) + possibly
  one LLM call for the question. Expect ~2–8s per chunk. First call after a cold GPU may be slower.
- **`/measure` latency:** ~1–2 min (many LLM calls). Block with a spinner or run async + poll.
- **Graceful degradation:** if the GPU ml-service is unreachable, audio endpoints return a *neutral*
  analysis (`confidence: 1.0`, no anomalies → `asked: false`) instead of erroring. Check
  `/confusion/health` to know whether the real engine is actually running.
- **`session_id`:** any string you choose. In-memory — gone on restart. One per teaching session.
- **`chunk_id`:** an integer you assign per chunk (increment it). It identifies the chunk for
  analysis storage and Q&A memory. If you want `/fusion` to line chunks up with transcript
  segments, use `chunk_id == Segment.id` (the same numbering).
- **`asked` is the only flag you need to branch on** in Flow A: render the question when `true`,
  skip when `false`.
```
