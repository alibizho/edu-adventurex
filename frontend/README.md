# Frontend — wut

The React UI. A pixel-art classroom where the kid states a topic (or uploads material), gets a
generated growth path, and then **teaches** each class out loud to an AI student that interrupts
with questions when it gets confused.

Stack: **React 19 + Vite 7 + react-router 7 + TypeScript**, `mermaid` for the concept map,
`react-markdown` + `remark-gfm` for teacher's notes, `lucide-react` icons, `@fontsource` for
Bebas Neue / Space Mono. No CSS framework — hand-written CSS in `src/styles/`.

## Run it

```bash
npm install
npm run dev        # http://127.0.0.1:5173
```

The backend must be up on `http://127.0.0.1:8000` (`cd ../backend && uvicorn app.main:app --reload`).
That origin is in the backend's `CORS_ORIGINS` default, so no extra config is needed.

```bash
npm run build      # tsc -b && vite build  -> dist/
npm run preview
```

## Environment

Copy `.env.example` to `.env`. Both variables are optional.

| Variable | Default | What it does |
|---|---|---|
| `VITE_API_BASE_URL` | `http://127.0.0.1:8000` | Backend origin. |
| `VITE_DATA_MODE` | `backend` | Reserved for forcing the static-mock screens. |

**Never put an LLM API key in a `VITE_*` variable** — Vite inlines those into the browser bundle.
Keys live server-side in `backend/.env` only.

## Routes

`src/app/routes.ts`

| Path | Page | Purpose |
|---|---|---|
| `/` | `MaterialPage` | Topic input + file upload → `/materials/extract` → `/plan/scope` → `/plan/build`. |
| `/concepts` | `ConceptPage` | The generated classes, as a mermaid map. Pick one to teach. |
| `/study` | `StudyPage` | The teaching session: notes, AI-student sidebar, mic. |
| `/summary` | `SummaryPage` | Post-class transfer-delta result (polls the analysis job). |
| `/map` | `KnowledgeMapPage` | Every growth path, and the way into any class. |
| `*` | `NotFoundPage` | |

## Talking to the backend

**Every** backend call is declared in one file:
`src/features/learning-data/backendLearningDataSource.ts`.
Add endpoints there, not inline in components.

- `apiClient.ts` — `fetch` wrapper: base URL, JSON/FormData handling, per-call timeout via
  `AbortController`, and an `ApiError` that surfaces the backend's `detail` string. The backend
  returns human-readable upper-case messages for LLM failures, so those reach the screen as-is.
- `backend.types.ts` — TypeScript mirrors of `backend/app/schemas.py`. Keep the two in sync.
- A class's session id is `` `${path_id}:${class_id}` `` (`classSessionId()`), which is what
  `/analysis/{session_id}` and `/sessions/{session_id}` key off.

Browser flow, end to end:

1. `POST /materials/extract` — uploaded PDFs/images/text extracted in memory (never persisted).
2. `POST /plan/scope` — confirm the topic, or pick one of 3 narrower options.
3. `POST /plan/build` — persists the growth path. `GET /plan`, `GET /plan/{id}` and
   `GET /plan/{id}/memory` restore the UI after a refresh.
4. `POST /plan/{id}/class/{cid}/notes` — teacher's notes; written by the build, so this only
   spends a call for an older path or with `?regenerate=true`.
5. Teaching happens in the live classroom. You arm the mic once and talk; `useContinuousRecorder`
   closes an utterance after a 3 s pause and posts it to `.../teach/audio-turn` with `silent=true`,
   so the chunk is transcribed and analyzed but nobody interrupts unless they have a real question.
   Chunks queue and drain one at a time — you can keep talking through the upload.
   When a question comes back, a `?` rises over one of the six seats (`classroom.seats.ts`).
   Clicking it zooms to that student, where your answer goes up with `silent=false` and is recorded
   against the question via `POST /questions/answer`.
   `degraded: true` means the GPU ml-service is unreachable — the room falls back to typing rather
   than pretending the turn happened.
6. `POST /plan/{id}/class/{cid}/end` — idempotent; folds the class into cross-class memory.
7. `POST /analysis/{session_id}` starts the transfer-delta measurement; `SummaryPage` polls
   `GET /analysis/{session_id}` until it is `complete` or `failed`.

## Layout

```
src/
├── app/            router, ScrollToTop, SessionProvider (client-side session state)
├── components/     layout chrome (AppHeader, StatusBar) + pixel-art visuals
├── features/       one folder per screen; *.data.ts holds static copy, *.types.ts the shapes
│   └── learning-data/   the entire backend boundary
└── styles/         one stylesheet per feature, all imported from index.css
```

`useWavRecorder.ts` captures mic audio and encodes 16-bit PCM WAV in the browser — that is the
format `POST .../teach/audio-turn` forwards to the ml-service.
