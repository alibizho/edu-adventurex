# wut

A kid teaches an AI that knows nothing — and we **measure** how much actually got through, by
experiment instead of a rubric. The AI plays a confused student (it only asks, restates, or admits
confusion); the kid is the teacher. After the lesson the AI sits a hidden test, and we compare
students who heard the lesson against a control that didn't. That gap is the real measure of
understanding. Cross it with how unsure the kid *sounded* while teaching, and a confident-but-wrong
explanation becomes a visible **blind spot** — the thing no rubric can catch.

## Folders

| Folder | What it is |
|---|---|
| `frontend/` | React + Vite UI (`:5173`). The pixel-art classroom: upload material → confirm the topic → build a plan → teach a class out loud → read the result. [Setup + route map](frontend/README.md) |
| `backend/` | FastAPI service (`:8000`). Plans and teacher's notes, the live teaching turn, objective mastery, transfer-delta measurement, cross-class memory. [HTTP reference](backend/API.md) · [teaching flow](backend/LEARN_BY_TEACHING.md) |
| `ml-service/` | GPU confusion engine (`:8100`), deployed on Hyper AI. Whisper ASR + Wav2Vec2/mDeBERTa + a judge LLM → per-utterance confidence. [Deploy guide](ml-service/README.md) |
| `data/` | Local scratch for models and fixtures. Contents gitignored. |

Three processes. The frontend talks only to the backend; only the backend talks to the ml-service,
and it degrades gracefully when that box is down.

## Run it

```bash
# 1. database
docker compose up -d db                       # Postgres on :5432

# 2. backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                          # add your LLM keys + ML_SERVICE_URL
uvicorn app.main:app --reload --port 8000     # docs at /docs

# 3. frontend (new terminal)
cd frontend && npm install && npm run dev     # http://127.0.0.1:5173
```

Notes:

- `.env.example` ships `STORE_BACKEND=db`, so plans, notes and progress survive a restart. Set
  `STORE_BACKEND=memory` to run with no database — but every class is then regenerated, and
  re-billed, after each reload. Tables are created on startup; there is no migration step.
- The **ml-service** runs on a GPU box, not locally. Point the backend at it with `ML_SERVICE_URL`.
  Without it, voice teaching returns `degraded: true` and the UI falls back to typing.
- PDF/image uploads need Tesseract (`brew install tesseract`). Text and Markdown work without it.
- Everything in one container instead: `docker compose up --build`.

## How the pieces fit

1. `POST /materials/extract` — pull text out of uploaded PDFs, images or notes, in memory.
2. `POST /plan/scope` → `POST /plan/build` — confirm or narrow the topic, then break it into ~5
   classes, each with 3–5 **objectives** the learner must be able to explain out loud.
3. `POST /plan/{id}/class/{cid}/notes` — a Markdown primer for the class, generated once and reused.
4. Teaching: every pause ships one utterance to `.../teach/audio-turn`. It is transcribed, scored
   for confusion, and checked against the objectives. The class stays quiet unless it has something
   to ask — either about something that sounded shaky, or about a goal you haven't covered yet.
5. `POST /plan/{id}/class/{cid}/end` — folds the class into cross-class memory so later classes
   don't re-teach or re-ask.
6. `POST /analysis/{session_id}` — the taught-vs-cold ensemble (transfer delta), crossed with how
   confident the speech sounded into `blind_spot` / `aware_gap` / `productive_struggle` / `mastery`.

## Contributing

**Setup** — follow *Run it* above, then `pip install -r backend/requirements-dev.txt`.

**Before you push:**

```bash
cd backend && source .venv/bin/activate && pytest -q     # 54 tests, hermetic: no LLM, no GPU, no DB
cd frontend && npm run build                             # tsc -b + vite build
```

**Where things go:**

- Backend routes stay thin. Orchestration lives in `app/curriculum/` (plans, teaching, mastery),
  `app/agents/` (the student and question writers) and `app/pipeline/` (measurement).
- Every frontend backend call is declared in
  `frontend/src/features/learning-data/backendLearningDataSource.ts` — add endpoints there, not
  inline in components. `backend.types.ts` mirrors `backend/app/schemas.py`; change both together.
- The backend and ml-service deploy separately and share no package, so
  `backend/tests/test_ml_service_contract.py` asserts their schemas agree. It runs on CPU with no
  torch — keep it that way.

**Conventions:**

- Tests are hermetic. Stub the LLM (see `_stub_llm` in `tests/test_workflow.py`) rather than
  calling one; a test that needs network or a GPU doesn't belong in the suite.
- Never put a key in a `VITE_*` variable — Vite inlines those into the browser bundle. Secrets live
  in `backend/.env`, which is gitignored along with anything matching `.env.*`.
- Tuning knobs go in `backend/app/config.py` with a default and a comment explaining the trade-off,
  so they can be moved from `.env` without a code change.

## Status

Running end-to-end against a live ml-service on Hyper AI and DeepSeek LLMs.

Known rough edge: the deployed GPU box predates the current `ml-service/` code. Until it is
redeployed, word-level timings come back degenerate (`/health` reports `pace_degraded`), so
hesitation is caught by the browser-measured prosody signal and a lexical backstop rather than by
the acoustic model. `ABSOLUTE_DISSONANCE` in `ml-service/config.py` is uncalibrated and wants
tuning against two real recordings — one confident, one hesitant.
