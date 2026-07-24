# Teachable Student

A kid teaches an AI that knows nothing — and we **measure** how much actually got through, by
experiment instead of a rubric. The AI plays a confused student (it only asks, restates, or admits
confusion); the kid is the teacher. After the lesson the AI sits a hidden test, and we compare
students who heard the lesson against a control that didn't. That gap is the real measure of
understanding. Cross it with how unsure the kid *sounded* while teaching, and a confident-but-wrong
explanation becomes a visible **blind spot** — the thing no rubric can catch.

## What's here

```
backend/     FastAPI service: teaching loop, transfer-delta measurement, fusion, and real-time
             per-chunk questions. Calls the ml-service for speech-confusion analysis.
             HTTP contract + frontend guide: backend/API.md
ml-service/  GPU confusion engine (Whisper ASR + Wav2Vec2/mDeBERTa/BGE + a judge LLM), deployed
             on Hyper AI. Setup: ml-service/README.md
frontend/    React UI (placeholder — not started).
```

## Run

**Backend** (local):
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # fill in LLM keys + ML_SERVICE_URL
uvicorn app.main:app --port 8000
# State is in-memory by default (lost on restart). For durable Postgres storage, set
# STORE_BACKEND=db + DATABASE_URL in .env — or just use Docker (below).
```
Interactive docs: http://localhost:8000/docs. Full endpoint reference: [backend/API.md](backend/API.md).

**ml-service** (Hyper AI GPU box): see [ml-service/README.md](ml-service/README.md). It must be
running for the audio confusion endpoints (`/confusion/analyze`, `/questions/from_chunk`); point the
backend at it with `ML_SERVICE_URL` in `backend/.env`.

**Docker** (Postgres + backend, durable storage): put your LLM keys + the Hyper AI `ML_SERVICE_URL`
in `backend/.env`, then:
```bash
docker compose up --build      # http://localhost:8000/docs
```
Runs `STORE_BACKEND=db`, so session context survives restarts. Inspect the DB with
`docker compose exec db psql -U ts -d teachable`.

## How it fits together

- **Real-time spoken class (primary flow):** the kid states a topic and teaches out loud. The
  frontend sends each paused speech chunk to `POST /questions/from_chunk`; the ml-service analyzes
  it and the backend generates a question only when the chunk sounded confused.
- **Measurement flow:** `POST /teach/turn` builds a transcript → `POST /measure` runs the
  taught-vs-cold ensemble (transfer delta) → `GET /fusion/{id}` crosses disturbance × delta into
  per-segment quadrants (`blind_spot` / `aware_gap` / `productive_struggle` / `mastery`).

## Status

Running end-to-end against a live ml-service on Hyper AI and DeepSeek LLMs. The ml-service's
confusion signals are still being tuned (its anomaly judges are noisy on short utterances), so the
real-time question gate currently fires on low confidence + a lexical hesitation backstop rather
than the raw anomaly flags — see [backend/API.md](backend/API.md) "Practical notes".