# Teachable Student — Technical Report (dev)

> Dev-team summary of the v2 architecture. The full spec is `newTR.md`; this is the
> scannable version for dividing up the build. Where they disagree, `newTR.md` wins.

## 1. What we're building

A kid teaches a topic to an AI that knows nothing, and we **measure** how well they actually
understood it — with an experiment, not a rubric. Three independent instruments read the same
hidden variable (real understanding) through different channels. Their agreement is the signal;
their disagreement is the diagnosis.

The AI student is constrained to only **ask, restate, or admit confusion** — it never explains
or answers. That constraint is what keeps the measurement honest.

## 2. Architecture at a glance

```
                        SOURCE DOC (what the kid studied)
                                   │
  kid speaks ──▶ STT ──▶ TRANSCRIPT (segments + clauses, shared IDs) ──▶ student agent replies
                                   │
        ┌──────────────────────────┼───────────────────────────┐
        ▼                          ▼                            ▼
  A. Transfer Delta          B. Speech Disturbance        C. Live Content Check
  (post-session test)        (live, from voice)           (live, per clause)
        │                          │                            │
        └──────────────────────────┴───────────────────────────┘
                                   ▼
                          FUSION (2×2 matrix + calibration ρ)
                                   ▼
                          Results screen / gap ledger
```

Everything keys off **shared segment IDs** (§7). Break that contract and the instruments stop
lining up.

## 3. Instrument A — Transfer Delta (primary)

**Question:** did the explanation actually transmit? This is the only channel with mechanical
ground truth and a control arm, so it's the one number we defend.

**Pipeline:**
1. **Generate** 15–20 *transfer* questions (novel scenarios, not recall) from transcript +
   source, using the strong model. **Each question ships with a machine-checkable answer key**
   (MCQ / numeric / constrained) — this is what makes grading mechanical.
2. **Filter** (load-bearing): run each candidate past a *cold* student (no transcript), k=3
   samples. Drop any the cold student gets right ≥2/3 — those measure priors, not teaching.
   Gate: if <25% survive, regenerate harder.
3. **Score ensemble:** 20 taught personas (transcript in context) + 10 cold personas (fresh
   samples, no transcript) answer the survivors closed-book. Personas vary by system-prompt
   seed + temperature, not by vendor.
4. **Delta:** `Δ_q = taught_mean − cold_mean`. Negative Δ = teaching made the learner *worse*.
   Report `Δ ± CI` (bootstrap over questions, not personas — personas are correlated).
5. **Attribution:** leave-one-out ablation on failed questions → "here's the sentence that
   broke it." Citations are a cheap prior; ablation is the real estimator.

**Model:** small/fast student model = **GLM-4.5-Air**; strong generator/verifier = **GLM-4.6**
(via z.ai, OpenAI-compatible). Small student is deliberate — a frontier model already knows the
syllabus and the delta collapses.

## 4. Instrument B — Speech Disturbance (live)

**Question:** did the kid *sound* like they understood? Runs on raw audio in parallel.

**Signals (3 tiers):** disfluencies (filled pauses, self-repairs, restarts), prosody (pitch
resets, uptalk, rate/energy dips, jitter/shimmer), lexical hedging ("maybe", "I think", "sort of").

**Models / tools (ship cheapest first):**
| Tier | Approach | Tooling |
|---|---|---|
| Baseline (day 1) | eGeMAPS + hedge-density + filled-pause → LightGBM/logreg → Dₛ | **openSMILE**, librosa/parselmouth |
| Upgrade | frozen **wav2vec2/HuBERT** encoder → small head, BIO disfluency tagging | HF transformers |
| Alignment | word/clause boundaries (feeds B *and* C) | **Whisper** word timestamps / MFA |

**Non-negotiable:** per-speaker z-normalization against the kid's own rolling baseline —
otherwise we just flag shy kids. Output = `Dₛ ∈ [0,1]` per segment + feature breakdown.

## 5. Instrument C — Live Content Check (live, optional)

**Question:** does the content hold together right now? Per completed clause:
1. **Claim gate** — is this an assertion? Skip questions/fragments.
2. **Clean** — strip disfluencies (reuses B's tagger; NLI on raw speech is out-of-domain).
3. **Retrieve** — top-3 prior claims + top-3 source passages (**BGE-M3 / bge-small** + FAISS).
4. **Verify with NLI cross-encoder:**
   - `C1` self-contradiction (clause vs earlier claim)
   - `C2` source-contradiction (clause vs source)
   - `C3` unsupported (log only)

**Model:** **DeBERTa-v3 MNLI cross-encoder** (or an LLM-per-clause call to ship day 1).
**Key design choice:** retrieval uses embeddings, verification uses a *cross-encoder* — cosine
distance can't tell "same topic" from "opposite claim" (see `newTR.md` Appendix A).

**Precision over recall:** false interrupts break the student persona. Target ≥0.85 precision,
≤1 false interrupt / 5 min. Arbitration + cooldown so the AI stays a student, not a fact-checker.

**Measurement vs tutoring mode:** in measurement mode C2 is logged but *not voiced* (voicing it
would hint the answer and contaminate Δ). Demo flow = attempt 1 measurement, attempt 2 tutoring.

## 6. Fusion & output

Cross **Dₛ** (sounded sure) × **Δₛ** (transmitted):

| | Failed (Δₛ ≤ 0) | Passed (Δₛ > 0) |
|---|---|---|
| **High Dₛ** | Aware gap — best learning target | Productive struggle — reinforce |
| **Low Dₛ** | **Blind spot — confident + wrong. The dangerous one.** | Mastery |

- **Calibration ρ** = Spearman(Dₛ, gap) — does the kid know what they don't know?
- **Live detection rate** = did B or C catch a blind-spot segment while the kid was still talking?

## 7. Data model (key entities)

```
session(id, child_id, topic, source_ref, mode)
source(id, ref, text, chunk_embeddings)
segment(id, session_id, idx, text, text_clean, t_start, t_end, audio_ref)   # the spine
clause(id, segment_id, idx, text_clean, is_claim)
question(id, session_id, text, format, answer_key, cold_pass_rate, survived)
score(question_id, arm['taught'|'cold'], persona_seed, correct, not_covered, cited_segment_ids[])
disturbance(segment_id, score, features)                # Instrument B
content_flag(clause_id, type, score, evidence, voiced)  # Instrument C
run(id, session_id, delta_overall, delta_ci_low/high, survival_rate, calibration_rho)
gap_ledger(id, child_id, question_id, source_ref, first_failed_at, reacquired_at)
```
Store: SQLite for the build, Postgres + JSONB (+ pgvector or in-memory FAISS) later.

## 8. Stack & models

| Layer | Choice |
|---|---|
| Backend | FastAPI, async (`asyncio.gather` + semaphore for the fan-out) |
| Student model | **GLM-4.5-Air** (small = sensitive delta) |
| Generator / verifier | **GLM-4.6** |
| Speech features | **openSMILE** (eGeMAPS), librosa / parselmouth |
| Disturbance model | **LightGBM** baseline → **wav2vec2 / HuBERT** head |
| Forced alignment | **Whisper** word timestamps / MFA |
| Retrieval | **BGE-M3 / bge-small** + FAISS or pgvector |
| Entailment (C) | LLM call → **DeBERTa-v3 MNLI** cross-encoder |
| STT / TTS | Browser Web Speech API (Whisper fallback) / `speechSynthesis` |
| Frontend | React, Tailwind, canvas sprite, live meters |
| Store | SQLite → Postgres + JSONB |

## 9. Main features

- Voice teaching loop with an AI student that only asks / restates / admits confusion
- Source-seeded sessions (topic picker with ~10 curated sources)
- Transfer-question generation **with machine-checkable answer keys**
- Cold-student filter + survival gate
- 20-taught / 10-cold ensemble → per-question + overall delta with bootstrap CI
- **Negative-delta detection** ("you made your classmate worse")
- Leave-one-out attribution → the exact sentence that broke
- Live speech-disturbance meter (per-speaker calibrated)
- Live content check (self- and source-contradiction) with measurement/tutoring modes
- Fusion 2×2 + calibration ρ + live-detection rate
- Gap ledger: failed questions return in later sessions → verified acquisition + delta curve

## 10. Scope & build order

Each step is gated — don't proceed past a failure.

1. **Real grader first** — answer keys + mechanical grading. Highest leverage; makes Δ real. *(this is where the current code is stubbed today)*
2. Filter harness go/no-go: good vs deliberately-bad transcript, prove Δ separates them
3. Ensemble fixes: 10 cold personas, fresh samples, bootstrap CI
4. Instrument B baseline: prove Dₛ higher on the bad transcript
5. Teaching loop + fusion results screen (the demo)
6. Instrument C, gap ledger, tutoring mode — upside, first to cut

**MVP = A (real) + B (baseline).** C is the designated cut. **Never cut** the cold arm, the
survival gate, or per-speaker normalization.

## 11. Ownership (4 people)

- **ML-transfer:** question gen + answer keys, filter, persona seeds, attribution, delta/CI math
- **ML-speech:** feature pipeline, disturbance model, per-speaker calibration, disfluency tagger; also owns C's threshold + eval set
- **Backend:** orchestration/fan-out, the segment/clause contract, retrieval index, C plumbing, interrupt arbitrator, store, fusion
- **UI:** voice loop, sprite, live meters, results screen (the results screen *is* the demo)

## 12. Current status (as of this writing)

The code is a partial Instrument-A skeleton with its core stubbed: the grader matches empty
references (everything counts correct), Instrument B returns 0.0, and C / source docs / fusion
don't exist yet. Step 1 above (the real grader) is the immediate next build.