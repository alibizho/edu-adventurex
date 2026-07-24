# Teachable Student

**A kid teaches an AI that knows nothing. We measure how much actually got through.**

Most AI tutors answer questions. A fluent answer *feels* like learning while nothing is learned. We flip it: the AI is the confused student, the kid is the teacher, and we measure real understanding two ways at once.

## The idea

1. **Kid teaches out loud.** The AI plays a student — it only asks, restates, or admits confusion. It never explains or answers its own questions.
2. **We test what transmitted.** The AI sits a hidden test. Two arms take it: one that heard the kid, one that heard nothing. The score is the **gap** between them. A negative gap means the kid taught something confidently wrong.
3. **We listen to *how* they said it.** A speech model flags uncertainty in real time — filled pauses, self-corrections, hedging, shaky pitch.
4. **We cross the two.** Fluent but wrong = a blind spot no rubric can catch. That's the whole product.

Full detail: [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md).

## Stack

- **Backend:** FastAPI (async), Postgres
- **Models:** small fast LLM for the student ensemble, larger LLM for question generation
- **Speech:** openSMILE + Whisper, LightGBM disturbance model
- **Frontend:** React + Tailwind, canvas sprite

## Team (4, moving fast)

| Owner | Area |
|---|---|
| ML–transfer | question generation, the cold-student filter, persona seeds |
| ML–speech | disturbance model, per-speaker calibration, eval |
| Backend | parallel scoring, transcript + gap store, delta math, fusion |
| UI | voice loop, sprite, results screen (this **is** the demo) |

## Contributing

- **Branch off `main`**, name it `area/thing` (e.g. `speech/calibration`). Small PRs, merge fast.
- **Keep the shared contract stable.** Everything keys off transcript **segment IDs** — don't change that schema without a heads-up in the group chat.
- **Two go/no-go experiments come first** (see report §11): (1) prove the delta separates a good vs bad transcript; (2) prove the speech model scores the bad one higher. 

## Run (WIP)

```bash
# backend
uvicorn app.main:app --reload
# frontend
npm install && npm run dev
```
