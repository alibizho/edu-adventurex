# Backend — Teachable Student

FastAPI service. Runs the teaching loop, the transfer-delta measurement, and the fusion.

## Layout

```
app/
  main.py          FastAPI entrypoint (CORS + routes)
  config.py        env-based settings
  llm.py           async OpenAI-compatible client (student + generator)
  schemas.py       shared data model — segments, questions, scores, runs
  agents/          <-- the agent workstream lives here
    prompts.py     system prompts (student, cold, generator) + persona seeds
    student.py     in-character student turn (only asks / restates / admits confusion)
    generator.py   transfer-question generation
    personas.py    ensemble persona spawning
  pipeline/        measurement
    filter.py      cold-student filter (k=3, drop what priors already answer)
    scoring.py     ensemble fan-out + delta math (asyncio.gather + semaphore)
    attribution.py map failed questions back to segments via citations
  speech/          Instrument B (ML-speech owner) — stub interface for now
  store/           persistence — in-memory for now, Postgres later
  api/routes.py    HTTP endpoints
```

## Run

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in keys
uvicorn app.main:app --reload
```

Open http://localhost:8000/docs for the live API.

## Where to start (agent work)

1. `agents/prompts.py` — tighten the student constraint and the persona seeds.
2. `agents/student.py` / `agents/generator.py` — wire the real LLM calls.
3. `pipeline/filter.py` + `pipeline/scoring.py` — the go/no-go experiment (report §11).

Everything returns mock data until wired, so the frontend can integrate against `/docs` today.
