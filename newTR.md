# Teachable Student: Technical Report

**v2. Merged from two drafts: the transfer-delta / speech-disturbance spec and the tri-modal contrastive confusion engine.**

> Learning by teaching, measured. A child teaches an AI that knows nothing. The system measures how much of the explanation actually transmitted, listens to how the child's speech betrays uncertainty while they talk, and checks live whether what they are saying contradicts themselves or the source. Three instruments, one latent variable: real understanding.

---

## 0. What changed in v2, and why

The two drafts were solving overlapping problems with different tools. This version keeps one spine and absorbs the other draft's best idea in a form that will actually work.

| Decision | Rationale |
|---|---|
| Transfer delta stays the primary measurement | It is the only channel with mechanical ground truth and a control arm. Everything else is a predictor of it. |
| Speech disturbance stays, unchanged in substance | Interpretable prosodic and lexical features with decades of psycholinguistic grounding, per speaker normalized. |
| **New: Instrument C, live content check** | The tri-modal draft was right that logic and fact errors should be caught *during* the session, not only after. That is the trigger the student agent needs in order to interrupt on content rather than only on hesitation. |
| **Instrument C is retrieval plus NLI, not cosine distance** | Cosine distance between bi-encoder embeddings conflates "about the same thing" with "says the same thing." Contradiction and argument reversal barely move a sentence embedding. See Appendix A. |
| **New: sessions are seeded with a source document** | Unlocks C2 (source contradiction), sharpens transfer question generation, and gives the gap ledger something to point at. This is the biggest product change in v2. |
| **New: measurement mode vs tutoring mode** | Interrupting on content risks teaching the child mid session and contaminating the delta. Modes make the tradeoff explicit and testable. See §6.4. |
| Tri-modal contrastive alignment is **not adopted** | Three concrete failure modes, documented in Appendix A so the reasoning is on record rather than relitigated. |

---

## 1. Problem

Every AI education product answers questions. A fluent explanation *feels* like learning while nothing is learned, and nobody can measure the gap. The field grades with rubrics: a language model judging a child's work on qualities it cannot observe, producing a number no one can defend.

We invert the roles and replace the rubric with an **experiment**, then instrument the child's speech and content while the experiment runs. The result is three measurements of the same hidden quantity, taken through channels that fail in different ways. When they agree, we trust the reading. When they disagree, the disagreement is the most useful signal in the product.

**Thesis:** understanding is not scored, it is *measured*. Once by what the child transmits, once by how the child sounds while transmitting it, once by whether the content holds together against itself and against the source.

---

## 2. System in one paragraph

The AI is a student that knows nothing about the topic. The child teaches it out loud from a source they studied. The AI asks the questions a confused classmate asks, says so when it does not understand, and never answers its own question. While the child talks, two live channels run: a speech model watching for the acoustic and lexical fingerprints of uncertainty, and a content checker testing each completed clause against the child's earlier claims and against the source passage. Afterwards the AI sits a test the child never sees. Two arms take that test: one that heard the child, one that heard nothing. The **transfer delta** is the gap between them. We cross that delta, segment by segment, against the live signals. Where the child sounded fluent, said nothing self contradictory, and the lesson still failed to transmit, we have found an illusion of knowing. That is the thing no rubric can see.

---

## 3. Three instruments

| | **A. Transfer delta** | **B. Speech disturbance** | **C. Live content check** |
|---|---|---|---|
| Question it answers | Did the explanation *transmit*? | Did the child *sound* like they understood? | Does the content *hold together* right now? |
| Channel | What reached a naive learner | Voice and word choice | Clause semantics vs prior claims and source |
| Latency | Post session, under 15 s | Streaming, under 300 ms | Streaming, under 600 ms per clause |
| Output | Per segment delta Δₛ ∈ [−1, 1] | Per segment disturbance Dₛ ∈ [0, 1] | Per clause flag Cₛ with type and evidence |
| Catches | Confident but wrong, gaps, misconceptions | Uncertainty, cognitive load, self doubt | Self contradiction, factual error, unsupported claim |
| Ground truth | Mechanical pass or fail of held out questions | Prosodic, lexical and disfluency evidence | NLI entailment against retrieved text |
| Misses | Nothing live | Content correctness | Omissions, and anything not stated as a claim |

Independence is what makes agreement meaningful and disagreement diagnostic. A never hears audio. B never sees the test or the source. C never sees the test. The only shared dependency is the segmentation and alignment layer, which is exactly why §7 treats it as the contract.

---

## 4. Instrument A: Transfer delta

### 4.1 Session setup

A session is seeded with a **source document**: the passage, page or article the child studied. This is new in v2 and it is load bearing in three places. It gives C2 something to check against, it lets the question generator ground counterfactuals in real content rather than only in the child's transcript, and it gives the gap ledger a target to point the child back to.

Practically: a topic picker with about ten precurated sources for the demo, freeform upload later.

### 4.2 Teaching session

Child speaks → STT → a **student agent** responds in character: confused, literal, missing prerequisites. Hard constraint in the system prompt: it may only **ask, restate, or admit confusion**. It may not explain, confirm correctness, or answer its own question. Any deviation is a bug, not a style choice.

The transcript is stored as **numbered segments**, roughly one segment per child utterance, with clause level substructure for Instrument C. Segment IDs are the primary key everything attributes to.

### 4.3 Question generation

From the transcript plus the source, generate 15 to 20 candidate questions with a **stronger model**. These must be **transfer** questions, not recall: novel scenarios that only resolve if the child's specific explanation carried.

- Recall (rejected): *"What did I say about friction?"*
- Transfer (kept): *"If the surface were ice instead, what happens to the box, and why?"*

**Every question is generated together with a machine checkable answer key.** This is the fix for the biggest hole in v1. The generator emits, per question:

```jsonc
{
  "text": "If the surface were ice instead, what happens to the box and why?",
  "format": "mcq",              // "mcq" | "numeric" | "short_constrained"
  "options": ["slides further", "stops sooner", "no change"],
  "answer": "slides further",
  "required_reason_keys": ["less friction"],   // any-of match, lemmatized
  "cites_segments": [3, 4]
}
```

Scoring is then a string, numeric or key match. No model grades the child, and no model grades a persona's free text either. When a short constrained answer is unavoidable, an LLM verifier compares the persona's answer to the **answer key**, never to the child's transcript. That distinction is what keeps "mechanical" honest, and it is worth saying out loud to judges.

### 4.4 Question filtering, the load bearing step

Run every candidate past a **cold student** (no transcript, no source), **k = 3** samples each. Delete any question the cold student answers correctly in **≥ 2 of 3** samples. What survives can only be passed on what the child actually transmitted.

```
Q* = { q ∈ Q : cold_pass_rate(q, k=3) < 2/3 }
```

**Survival gate:** if `|Q*| / |Q| < 0.25`, the questions are measuring priors, not teaching. Regenerate harder. This gate is the single best early warning that the whole measurement has collapsed.

**Statistical trap, new in v2: do not reuse the filter samples as the cold arm at scoring time.** Questions were *selected* for low cold pass rate on those exact samples, so those samples are biased downward by the selection. Pooling them into `z_q` inflates Δ by the classic winner's curse. Draw **fresh** cold samples at scoring time. This costs about 50 extra calls and it is the difference between a real number and a flattering one.

### 4.5 Ensemble scoring

Spawn **P = 20 taught personas** with varied priors: missing a prerequisite, holding a common misconception, very literal, overgeneralizes, and so on. Variation comes from system prompt seeds plus temperature. Each gets the transcript in context and answers closed book: *answer only from the lesson, otherwise output* `NOT COVERED`.

Cold control arm at scoring time: **P′ = 10 personas, no transcript, fresh samples.** v1 specified 5. Ten costs little and matters, because `z_q` with five samples is quantized to 0.2 and the whole negative delta claim rests on resolving `z_q > t_q` precisely.

`NOT COVERED` counts as incorrect, but is recorded separately. It distinguishes a **gap** (the child never covered it) from an **error** (the child covered it and the learner got it wrong). Those route to different interventions.

### 4.6 Delta math

Correctness `c ∈ {0, 1}`, or graded `[0,1]` where the answer key supports partial credit. Graded is preferred: with this few samples, the extra resolution buys real power.

```
Taught mean, per question:   t_q = mean over p ∈ P    of c_taught[p, q]
Cold mean,   per question:   z_q = mean over p′ ∈ P′  of c_cold[p′, q]

Per question delta:          Δ_q = t_q − z_q
Overall delta:               Δ   = mean over q ∈ Q* of Δ_q
```

- `Δ_q > 0` → teaching helped.
- `Δ_q ≈ 0` → teaching was inert.
- **`Δ_q < 0` → teaching made the learner worse.** *"You made your classmate worse than if you had said nothing."* No rubric based tool can produce that moment.

**Uncertainty, new in v2.** Twenty personas seeded from one base model are not twenty independent learners. Their errors are correlated, so treating persona count as sample size overstates confidence badly. Bootstrap over **questions**, which are the closer to independent unit:

```
CI(Δ) = percentile bootstrap over q ∈ Q*, B = 2000 resamples
```

Report `Δ ± CI`. If the interval crosses zero, say so on the results screen instead of pretending. If two vendors are available, split the taught arm across base models and report the between model variance component. If not, note the limitation in one line rather than hiding it.

### 4.7 Attribution

For each failed question we need *"here is the sentence where it broke."*

- **Prior (cheap):** personas cite the segment IDs they relied on. Useful as a candidate set, but weak on its own. Models rationalize citations after the fact, so a citation is evidence about where the model looked, not about what caused the failure.
- **Estimator (do this):** **leave one out ablation**, restricted to failed questions and to the segments those questions cited. Rerun with one segment removed and see which removal flips the answer. Scope: typically 5 to 8 failed questions × 2 to 3 candidate segments × 3 personas, so roughly 50 extra calls. That is affordable, and it turns attribution from a claim into a measurement.

`Δ_s = mean of Δ_q over questions attributed to s`. Segments with no attributed questions have Δₛ **undefined**, not zero. Do not let undefined segments enter the fusion matrix as if they were mastered.

### 4.8 Persistence

A **gap ledger** of every question the child's teaching failed to carry, with a pointer back to the source passage that covers it. Failed questions return unannounced in later sessions. Answering one then is **verified acquisition**, timestamped. Session 3 vs session 1 on the same topic yields a **delta curve**, not a grade.

---

## 5. Instrument B: Speech disturbance

Runs on raw audio, in parallel with teaching. Asks: *where did the speaker's own production signal uncertainty, effort, or a mismatch between confidence and content?* Grounded in psycholinguistics. Disfluencies and prosodic disturbance correlate with cognitive load, lexical retrieval difficulty and metacognitive uncertainty.

### 5.1 Signal taxonomy

**Tier 1, disfluencies (production level)**
- Filled pauses: *um, uh, er*
- Silent pauses, thresholded. **Mid clause pauses weigh more than clause boundary pauses.**
- Repetitions (*"the the"*), false starts, restarts
- Self repairs (*"it goes up, no, down"*), the strongest single marker of a live correction
- Prolongations and segmental lengthening

**Tier 2, prosodic disturbance (delivery level)**
- F0 variability, pitch resets, and **rising terminal intonation on declaratives** (uptalk as uncertainty)
- Articulation rate dips
- Energy and loudness dips
- Jitter and shimmer, micro instability of the vocal source under stress

**Tier 3, lexical and semantic hedging (content level)**
- Epistemic hedges: *maybe, I think, kind of, probably, sort of*
- Approximators and vague reference: *this thing, that part, whatever*
- Discourse markers used as stalls: *like, you know*

The valuable signal is the **discrepancy**, not any single disturbance. High fluency on wrong content is dangerous; disfluency on correct content is healthy productive struggle. B alone cannot tell them apart. Fusion resolves it (§6).

### 5.2 Feature extraction

| Group | Features | Tooling |
|---|---|---|
| Acoustic, frame level | F0, RMS energy, MFCC 0 to 12, ZCR, spectral centroid and flux, jitter, shimmer @ 10 ms hop | openSMILE eGeMAPSv02 (88 dim), or librosa / parselmouth |
| Forced alignment | word and clause time boundaries | Whisper word timestamps, or Montreal Forced Aligner |
| Disfluency tokens | filled pause, repair and restart spans | disfluency aware ASR plus tagger |
| Lexical | hedge lexicon density, epistemic marker counts | lexicon lookup plus small classifier |

**Shared with Instrument C:** the forced aligner supplies C's clause boundaries, and the disfluency tagger supplies C's cleaned clause text. This is not incidental. NLI models are trained on clean written English and degrade sharply on raw disfluent child speech, so stripping *"um"*, repetitions and abandoned restarts before C sees the text is a correctness requirement, not an optimization.

### 5.3 Model tiers, cheapest first

1. **Baseline, ship this first.** openSMILE eGeMAPS plus hedge density plus filled pause rate → **LightGBM or logistic regression** → per segment graded score. Fast, interpretable, trains on tiny data, no GPU.
2. **Sequence model.** Frozen **wav2vec2 or HuBERT** encoder → small BiLSTM or Transformer head → frame level BIO tagging of disfluency spans. Fine tune the head only.
3. **End to end multitask.** Shared Whisper encoder, two heads: ASR with timestamps, and frame level disturbance classification.
4. **Late fusion, target.** Acoustic, prosody and lexical models combined into one **calibrated** probability per word, pooled to the segment.

```
Fusion:  D_word = σ( w_a·a + w_p·p + w_l·l + b )
Pool:    D_s    = pool_{word ∈ s}( D_word )     # mean for coverage, max for spikes
```

### 5.4 Training data and labels

- **Supervised corpora:** Switchboard disfluency and EDITED annotations, filled pause corpora, uncertainty annotated speech.
- **Zero training path, realistic for the build:** pretrained disfluency detectors plus prosody heuristics plus a hedge lexicon deliver most of the signal with no training.
- **Weak supervision, stretch:** use A's per segment failures as distant positive labels. **Guard against circularity.** Never let the same signal both train B and validate B. Hold out a clean split and keep primary validation intrinsic (§8).

### 5.5 Per speaker calibration, non negotiable

Every child has a personal disfluency rate. A naturally halting speaker is not "always uncertain." Normalize each feature against the child's own rolling baseline:

```
z_s(f) = ( f_s − μ_child(f) ) / σ_child(f)
```

`μ_child` and `σ_child` come from a short warm up utterance plus a rolling window. **We measure deviation from the child's own baseline, never absolute disfluency.** Without this the model just flags shy kids, which is both a correctness failure and an ethical one.

### 5.6 Runtime modes

- **Streaming, under 300 ms.** Lightweight prosody, filled pause and hedge detection during teaching. Drives the live meter, the confusion sprite, and one of the interrupt triggers.
- **Batch, post session.** Full fusion over recorded audio, producing the disturbance timeline and Dₛ per segment for the fusion matrix.

### 5.7 Output schema

```jsonc
{
  "segment_id": 4,
  "text": "...and then the box just kind of stops, I think",
  "t_start": 41.2, "t_end": 47.8,
  "disturbance": {
    "score": 0.81,                 // D_s ∈ [0,1], speaker normalized
    "filled_pause_rate": 0.12,
    "silent_pause_ratio": 0.34,    // mid clause weighted
    "repair_count": 1,
    "pitch_uncertainty": 0.72,
    "rate_dip": 0.28,
    "hedge_density": 0.20,
    "dominant_marker": "self_repair"
  }
}
```

---

## 6. Instrument C: Live content check

This is the salvaged core of the tri-modal draft, rebuilt on machinery that can actually make the distinction it needs to make. The original design asked cosine distance between bi-encoder embeddings to detect contradiction. It cannot: *"DNS translates domains to IPs"* and *"DNS translates IPs to domains"* share every content word and sit almost on top of each other in embedding space. Argument reversal is close to invisible to a distance metric. Full reasoning in Appendix A.

**The fix is a division of labour: embeddings retrieve, a cross encoder verifies.** A cross encoder attends jointly across both sentences and is trained on exactly the "same topic, opposite claim" distinction. That is what NLI models are for.

### 6.1 Pipeline

Per completed clause, boundaries supplied by the forced aligner:

1. **Claim gate.** Is this clause an assertion? Skip questions, meta talk, self addressed speech, and fragments. Cheap classifier or heuristic (declarative mood, contains a content noun and a finite verb). Precision here saves everything downstream.
2. **Clean.** Strip disfluency spans using B's tagger. NLI on raw *"the the box um kind of stops"* is out of domain and unreliable.
3. **Retrieve, two pools in parallel.**
   - Prior claims from this session, top 3 by embedding similarity.
   - Source document passages, top 3 by embedding similarity (bge-small or BGE-M3).
4. **Verify with NLI.** For each retrieved item, run a cross encoder over the pair.
   - `C1 self_contradiction`: contradiction between the current clause and a prior claim.
   - `C2 source_contradiction`: the source passage contradicts the clause.
   - `C3 unsupported`: no retrieved passage entails the clause and similarity is low. Weakest flag, log only, never interrupts.
5. **Emit** a flag with type, score, and the evidence span that triggered it.

```
C1: NLI(prior_claim  →  clause)  = contradiction
C2: NLI(source_span  →  clause)  = contradiction
C3: max_i entail(source_span_i → clause) < τ_support
```

### 6.2 Model choice and latency

| Option | Model | Latency per pair | When |
|---|---|---|---|
| Ship first | Single LLM call per clause with all six retrieved items, returning `{verdict, type, evidence}` as JSON | 300 to 800 ms | Day 1. Zero setup, good enough. |
| Upgrade | DeBERTa-v3 base or small, MNLI fine tuned, CPU | 20 to 40 ms | When false interrupt rate or latency demands it. Six pairs per clause fits inside 250 ms. |

Note the irony worth telling judges: the original draft picked DeBERTa-v3 as the text encoder, which is the right model. It just needed to be used as a cross encoder rather than a bi-encoder.

Clauses arrive every 3 to 5 seconds of speech, so even the LLM path is comfortably off the critical path.

### 6.3 Precision over recall

A false interruption is expensive. It breaks the student persona, derails the child, and pollutes the transcript. Tune the threshold to a **high precision operating point** and accept the recall loss.

Targets:
- Precision of interrupting flags ≥ 0.85 on the hand labelled dev set.
- **At most one false interrupt per 5 minutes** of speech.
- Every flag is logged regardless of threshold. Only flags above `τ_interrupt` are voiced.

### 6.4 Interrupt policy, and the contamination problem

Two channels can now request an interrupt: a B disturbance spike, and a C content flag. Without arbitration the AI becomes a twitchy fact checker and stops being a student.

**Arbitration:**
1. Priority: `C1 self_contradiction` > `C2 source_contradiction` > B spike > scheduled curiosity question.
2. Never interrupt mid clause. Queue to the next boundary.
3. Cooldown of 20 s after any interrupt.
4. Ceiling of one interrupt per 45 s of child speech.

**Voice.** A C triggered interrupt must be phrased **as confusion, never as correction**, because the moment the AI corrects, it stops being the naive student and Instrument A's premise collapses.

- Good: *"wait, I thought you said the box stops because of the surface? now it's the weight?"*
- Bad: *"actually, friction depends on the normal force."*

**Contamination.** C2 is the harder case: voicing a source contradiction, even as confusion, hints that the child is wrong. If the child then self corrects, the delta measures understanding *after a hint*, not unprompted understanding. So make it a mode, explicitly:

| Mode | C1 voiced | C2 voiced | Delta means |
|---|---|---|---|
| **Measurement** | yes | no, logged only | Unprompted understanding. Clean. |
| **Tutoring** | yes | yes, as confusion | Understanding after live scaffolding. Contaminated but pedagogically better. |

C1 is safe in both modes: pointing out that two of the child's own statements disagree uses only information the child supplied. It supplies no answer.

Default flow, which is also the demo: **first attempt in measurement mode, second attempt in tutoring mode.** Stretch: randomize mode across sessions and test whether tutoring mode improves the next session's delta more than measurement mode does. That is a real efficacy claim about the product itself, from the product's own instrumentation.

### 6.5 Output schema

```jsonc
{
  "clause_id": "4.2",
  "segment_id": 4,
  "t_start": 44.1, "t_end": 46.9,
  "text_clean": "the box stops because it is heavy",
  "flags": [
    {
      "type": "source_contradiction",
      "score": 0.91,
      "evidence": {
        "kind": "source",
        "ref": "src://friction_p2#s3",
        "text": "A heavier box experiences more friction, but also more inertia; on the same surface both boxes decelerate at the same rate."
      },
      "voiced": false,           // measurement mode
      "voiced_reason": "mode=measurement"
    }
  ]
}
```

### 6.6 What C cannot do

C sees only what was said. It cannot detect an **omission**, and omissions are a large share of failed transfer. A child who never mentions friction at all produces no contradiction and no unsupported claim, yet the lesson fails completely. Instrument A remains the only channel that catches that, which is precisely why A stays primary and C stays a live predictor rather than a replacement.

---

## 7. Shared contracts

Three instruments, one spine. Break the spine and nothing lines up.

- **Segment IDs** are the primary key. Roughly one per child utterance. Assigned once, by the transcript service, never recomputed downstream.
- **Clause IDs** are `segment.index`, assigned by the forced aligner. C emits per clause, fusion rolls up to segment.
- **Timestamps** all derive from the same aligner. Do not let B use Whisper timestamps and C use a different segmenter, or disturbance lands on the wrong sentence and the fusion matrix quietly becomes noise.
- **Cleaned text** is produced once by B's disfluency tagger and consumed by C. One implementation, one behaviour.
- Nobody changes this schema without a heads up in the group chat.

---

## 8. Fusion

### 8.1 Confidence × competence matrix

Cross **Dₛ** (how the child sounded, live) with **Δₛ** (whether it transmitted, measured). This 2 × 2 is the crown jewel of the product.

| | **Failed transfer (Δₛ ≤ 0)** | **Passed transfer (Δₛ > 0)** |
|---|---|---|
| **High disturbance (Dₛ ↑)** | **Aware gap.** Felt it, could not teach it. *Best learning target.* | **Productive struggle.** Uncertain but it landed. *Reinforce.* |
| **Low disturbance (Dₛ ↓)** | **Blind spot.** Confident and wrong. *Source of negative delta. Dangerous.* | **Mastery.** Fluent and it transmitted. |

### 8.2 Where C sits

C is a **live predictor of the bottom left cell**. Previously that cell was only visible after the session. Now, when the blind spot is an error rather than an omission, it is visible while the child is still talking.

New headline metric, **live detection rate**:

```
LDR = P( segment carried a live flag (B spike or C flag) | segment ended in the blind spot cell )
```

Decompose it. C should dominate on error type blind spots, B should be near chance on them by construction (the child sounded fine, that is what makes it a blind spot), and neither catches omissions. Reporting that decomposition honestly is more convincing than one inflated number.

### 8.3 Metacognitive calibration

Does the child know what they do not know?

```
gap_s = 1 − pass_rate_s                        # fraction of attributed questions that failed
ρ     = Spearman( D_s , gap_s ) over segments  # per session, and longitudinal
```

- **ρ high and positive** → well calibrated. Uncertain exactly where the lesson failed.
- **ρ near zero or negative** → poorly calibrated. Confident on the wrong things, the pattern that most predicts fragile learning.

Compute over segments with **defined** Δₛ only. With a 90 second session you may have 8 to 15 such segments, so ρ is noisy. Report it with a bootstrap interval and treat the longitudinal trend as the real product metric, not any single session's value.

### 8.4 Closing the loop

1. A live disturbance spike or content flag on segment s → the student agent asks its clarifying question *there*, subject to §6.4 arbitration, while the child is already primed to reconsider.
2. Post session, segments that were **low disturbance but failed** surface first: *"you sounded sure here, let's check it."*
3. Returning questions from the gap ledger are prioritized by quadrant. Blind spots before aware gaps.
4. Every blind spot links back to the source passage that covers it.

---

## 9. Worked example

Child is teaching why a heavy box slides less far than a light one. Source passage says both decelerate at the same rate on the same surface.

Utterance, segment 4:

> *"and then the box just... um... kind of stops, I think, because it's heavy"*

**Instrument B, live at t = 47.8 s**
- Filled pause plus 900 ms mid clause silence plus two hedges (*kind of*, *I think*).
- Rising terminal on a declarative.
- Speaker normalized `Dₛ = 0.81`, dominant marker `hedge_stall`.
- Above interrupt threshold, queued.

**Instrument C, live at t = 46.9 s**
- Clause gate: assertion. Cleaned to *"the box stops because it is heavy."*
- Retrieval pulls source span: heavier means more friction and more inertia, same deceleration.
- NLI verdict: **contradiction, 0.91**. Type `source_contradiction`.
- Mode is measurement, so `voiced = false`. Logged.
- C1 finds no prior claim conflict.

**Arbitration.** C2 outranks B but is muted in measurement mode, so the B spike is voiced. The student agent says: *"wait, sorry, I got lost. what makes it stop?"* No content is supplied.

**Instrument A, post session**
- Two surviving questions attribute to segment 4. Both fail.
- One fails *below* the cold arm: taught personas confidently answer "the heavy one stops sooner because weight makes friction stronger," while cold personas hedge and land closer to correct.
- `Δ_q = 0.15 − 0.60 = −0.45`. Leave one out ablation confirms removing segment 4 flips both.
- `Δ₄ = −0.45`.

**Fusion.** `Dₛ = 0.81`, `Δₛ < 0` → **aware gap**, not a blind spot. The child felt the shakiness and it showed. C independently confirmed the error live. This is the three way agreement case: two live channels and one post hoc measurement pointing at the same sentence for different reasons.

**Results screen.** Segment 4 in red, negative delta called out, the exact sentence quoted, the source passage shown beside it, and the question that flipped. Second attempt runs in tutoring mode, where C2 is now voiced as confusion at that same moment.

---

## 10. Data model

```
session(id, child_id, topic, source_ref, mode ENUM('measurement','tutoring'), created_at)
source(id, ref, title, text, chunk_embeddings)
segment(id, session_id, idx, text, text_clean, t_start, t_end, audio_ref)
clause(id, segment_id, idx, text_clean, t_start, t_end, is_claim BOOL)
question(id, session_id, text, format, options JSONB, answer_key JSONB,
         cold_pass_rate, survived BOOL)
score(question_id, arm ENUM('taught','cold'), persona_seed, correct,
      not_covered BOOL, cited_segment_ids[])
ablation(question_id, removed_segment_id, flipped BOOL)      -- Instrument A attribution
disturbance(segment_id, score, features JSONB)               -- Instrument B
content_flag(clause_id, type, score, evidence JSONB, voiced BOOL)  -- Instrument C
interrupt(id, session_id, clause_id, trigger, suppressed_by, at)
run(id, session_id, delta_overall, delta_ci_low, delta_ci_high,
    survival_rate, calibration_rho, live_detection_rate, created_at)
gap_ledger(id, child_id, question_id, source_ref, first_failed_at, reacquired_at NULL)
```

Postgres plus JSONB, or SQLite for speed of build. `source.chunk_embeddings` can be pgvector, or an in memory FAISS index given the corpus is a handful of documents.

---

## 11. Evaluation

A model that produces a number nobody validates is a rubric with extra steps. Every instrument gets its own eval.

**Instrument A**
- Filter survival rate per session, distribution across sessions.
- Δ separation between a deliberately good and deliberately bad transcript on the same topic. This is the go / no go.
- Bootstrap CI width on Δ. If the median session's CI crosses zero, the ensemble is too small or the questions too noisy.
- Attribution: agreement between cited segments and leave one out ablation. Low agreement is a finding worth reporting, not a bug to hide.

**Instrument B, intrinsic**
- Disfluency span detection precision, recall, **F1** against a held out Switchboard split.
- Binary disturbance classifier **ROC AUC**.
- **Expected Calibration Error** of Dₛ.

**Instrument C, intrinsic**
- Precision and recall of contradiction flags against about 100 hand labelled clauses from pilot sessions. Two annotators, report agreement.
- **False interrupt rate per minute** at the shipped threshold. This is the number that decides whether C is on or off in the demo.
- Claim gate precision. Garbage in here poisons everything after.

**Extrinsic, the money metrics**
- **AUC of Dₛ predicting per segment transfer failure (Δₛ ≤ 0).** If speech disturbance predicts transfer failure above chance, the two instrument thesis holds empirically.
- **Precision of C flags predicting Δₛ ≤ 0** on the same segment. C should be higher precision and lower recall than B here. If it is not, C is not earning its complexity.
- Live detection rate (§8.2), decomposed by channel and by failure type.
- Three way agreement table across A, B and C, and the ρ distribution across sessions.

**Ablations**
- B: acoustic only → plus prosody → plus lexical. Each tier should earn its place.
- B: per speaker normalization on vs off. Expect a large drop when off.
- C: retrieval plus NLI vs **cosine distance baseline**. Run this one. It is a two hour experiment that either validates the Appendix A argument with numbers or overturns it, and either outcome is worth having.

---

## 12. Stack

| Layer | Choice | Why |
|---|---|---|
| Backend | FastAPI, async | Known, and the fan out is IO bound |
| Ensemble | `asyncio.gather` plus semaphore | Plain concurrency beats a framework here |
| Session flow | LangGraph, optional | Only if the tutoring state machine grows. Skip if it slows you |
| Student model | Small fast model, Qwen or GLM tier | Sensitivity, cost, latency (§13) |
| Generator model | Larger model | Question quality is the bottleneck |
| Speech features | openSMILE, librosa, parselmouth | eGeMAPS plus prosody, CPU only |
| Disturbance model | LightGBM baseline → wav2vec2 or HuBERT head | Ship baseline day 1, upgrade if time |
| Forced alignment | Whisper word timestamps, or MFA | Serves B's attribution and C's clause boundaries |
| Retrieval | bge-small or BGE-M3 plus FAISS or pgvector | Corpus is tiny, either is fine |
| Entailment | LLM call day 1 → DeBERTa-v3 MNLI cross encoder | §6.2 |
| STT | Browser Web Speech API, Whisper fallback | Free, instant, also feeds the disfluency tagger |
| TTS | Browser `speechSynthesis` | Same |
| Frontend | React, Tailwind, canvas sprite | 8 bit face, live meters, results view |
| Store | Postgres plus JSONB, or SQLite | §10 |
| Deploy | Vercel plus a single backend host | Do this at hour 20, not hour 35 |

---

## 13. Latency and cost

**Model choice, the non obvious part.** Use a **small model as the student**. A frontier model already knows the syllabus, passes everything, and the delta collapses toward zero. A smaller model has more headroom, which means a more sensitive measurement, a cheaper ensemble, and 40 parallel calls fast enough for a live demo. Use a stronger model only for question generation and for explaining failures.

**Instrument A, roughly 200 calls per session:**
- Filtering: 15 to 20 candidates × k = 3 ≈ 50
- Taught arm: 20 personas × surviving questions, batched ≈ 20 to 100
- Cold arm, fresh samples: 10 personas × surviving questions ≈ 10 to 50
- Ablation: ≈ 50
- Generation and explanation: a handful

At concurrency 40 with a small model: **under 20 s end to end.** v1 claimed 15 s at 100 calls. The extra cost buys the fresh cold arm and real attribution, which are the two things that make the number defensible. Cut ablation first if you are over budget.

**Instrument B:**
- Streaming detector: under 300 ms per window, CPU, during teaching. Zero added wait at results time.
- Batch fusion: openSMILE plus alignment plus LightGBM over a 90 s clip ≈ 2 to 4 s, overlapping the ensemble. Off the critical path.

**Instrument C:**
- Retrieval: under 20 ms, index is tiny and in memory.
- Verification: 300 to 800 ms LLM path, 150 to 250 ms NLI path, per clause.
- Budget is a clause every 3 to 5 s, so both fit with room. Runs during teaching, adds nothing to results latency.

Net: the live demo returns in well under 25 seconds, and both live instruments cost nothing at results time because they run while the child is already talking.

---

## 14. Build order

Each numbered item is shippable and gated. Do not proceed past a failed gate.

1. **Filter harness, no UI.** Hardcode two transcripts on the same topic, one good and one deliberately bad. Prove Δ separates them. **Go / no go, first three hours.**
2. **Speech baseline, no UI.** openSMILE plus filled pause plus hedge density → Dₛ on the same two clips. Prove Dₛ is higher on the bad one. **Go / no go.**
3. **Content check, no UI.** Feed the bad transcript's clauses plus the source through retrieval and entailment. Prove it flags the planted error and does not flag the good transcript. **Go / no go. If this fails, cut C and ship two instruments.**
4. **Teaching loop.** Voice in, student agent that only asks, transcript segmentation with shared IDs.
5. **Two arm delta.** Taught vs cold, single persona, one number.
6. **Ensemble.** Same code in a loop across 20 personas plus 10 cold. The demo line: *worked on 14 of 20*.
7. **Live meters.** Stream Dₛ and C flags into the sprite. Wire the interrupt arbitrator.
8. **Attribution.** Citations first, leave one out ablation if time.
9. **Fusion view.** The 2 × 2, calibration ρ, negative delta in red, the exact sentence that lost each classmate, source passage alongside.
10. **Gap ledger and returning questions.** Can be seeded with fake history if time runs out.
11. **Later, not now:** generative practice panels, the stubborn classmate that holds a misconception and argues, the per speaker calibration warm up flow, mode randomization for the efficacy study.

**Cut order if behind:** ablation (8) → C (3, 7) → gap ledger (10) → drop to 10 taught personas. Never cut the cold arm, the survival gate, or per speaker normalization. Those three are the difference between a measurement and a demo.

---

## 15. Ownership

Four people. C adds load, so it is split rather than owned, and it is the designated cut.

- **ML transfer:** question generation with answer keys, the cold filter, persona seeds, attribution, delta math and CIs.
- **ML speech:** feature pipeline, disturbance model, per speaker calibration, disfluency tagger, the extrinsic eval. **Also owns the C threshold and the C eval set**, because it is the same precision and recall discipline.
- **Backend:** orchestration and fan out, the segment and clause contract, retrieval index, the C pipeline plumbing, interrupt arbitrator, store, fusion.
- **UI:** voice loop, sprite, live disturbance meter, live content flag surface, results screen. **The results screen is the demo.** Treat it as a first class deliverable, not a wrapper.

---

## 16. Risks

| Risk | Detection | Mitigation |
|---|---|---|
| Student model too smart, delta collapses | Filter survival rate under 25% | Smaller model, harder transfer questions, enforce `NOT COVERED` |
| Questions are recall not transfer | Cold arm scores high | Regenerate with explicit novel scenario instruction |
| Closed book leaks | Cold arm nonzero everywhere | Acceptable. Leakage is roughly symmetric across arms and subtracts out |
| **Cold arm reused from filter samples** | Δ suspiciously large, negative deltas rare | **Fresh cold samples at scoring time. Never reuse selection samples (§4.4)** |
| **Persona correlation inflates confidence** | CI implausibly tight | **Bootstrap over questions, not personas. Report the limitation** |
| **Attribution is model rationalization** | Citations disagree with ablation | **Ablation is the estimator, citations are the prior (§4.7)** |
| Voice latency kills the feel | Test on stage hardware early | Fall back to typed input for the demo |
| Ensemble too slow live | Time it at hour 10 | Precompute one session, run the second live |
| Speech model flags naturally disfluent kids | Dₛ high across a whole speaker | Per speaker z normalization (§5.5). Never absolute thresholds |
| Dₛ does not predict Δₛ | Extrinsic AUC near 0.5 | Fall back to A alone, keep Dₛ as a UI cue and not a claim |
| Circular training, B trained on A then validated on A | Suspiciously perfect agreement | Clean held out split, primary validation stays intrinsic |
| Forced alignment drifts | Disturbance lands on the wrong segment | Whisper word timestamps, sanity check against boundaries |
| **C false interrupts destroy the student persona** | False interrupt rate above 1 per 5 min | **High precision threshold, cooldown, ceiling (§6.3, §6.4)** |
| **C2 teaches the child and contaminates Δ** | Delta improves without the child improving | **Measurement mode mutes C2. Modes are logged per session (§6.4)** |
| **NLI out of domain on disfluent child speech** | Contradiction scores erratic on clean content | **Run NLI on B's cleaned clause text, never raw ASR (§5.2)** |
| **No source document for a freeform topic** | C2 has nothing to retrieve | Curated topics for the demo. C degrades to C1 only, which still works |
| **Claim gate over triggers on fragments** | Flags on non assertions | Tune gate precision first. It is upstream of everything in C |

---

## 17. Demo

Child teaches a topic **badly** for 90 seconds, in measurement mode. The live meter shows disturbance spikes as they hedge and self correct. A content flag lights quietly without interrupting, because measurement mode is honest. The score comes back low with one **negative delta** visible. The 2 × 2 lights up a **blind spot**, a place where the child sounded sure and was wrong, with the source passage shown next to their own sentence. Child reads the source, teaches again, this time in tutoring mode, where the AI now says *"wait, I thought you said..."* at exactly the moment the content breaks. Disturbance flattens, the delta jumps, calibration ρ climbs. The screen shows which classmates were won, which were lost, the exact sentence that lost them, and how the child's self knowledge improved between the two attempts.

---

## 18. Grounding

Learning by teaching is twenty years of published research, teachable agents and Betty's Brain out of Vanderbilt. Speech disfluency as a window into cognitive load and metacognitive uncertainty is decades of psycholinguistics. Textual entailment as a verification layer over retrieval is standard practice in fact checking and grounded generation.

What that combined literature never had is **honest, automated measurement**, because transfer scoring required human graders and disfluency analysis required hand annotation. The **control arm**, the **ensemble**, the **speech disturbance model**, the **live entailment check**, and the **fusion of all three into a calibration signal** are the contribution.

---

## Appendix A: Designs considered and not adopted

### A.1 Tri-modal contrastive alignment

The alternative draft proposed three frozen encoders (wav2vec2 for audio, DeBERTa-v3 for text, BGE-M3 for retrieval), trainable projection heads into a shared space, a cross attention layer for word level alignment, and confusion detected as cosine distance across three spaces: audio vs text, text vs text, text vs knowledge. No labels required.

The ambition was right and two of its ideas survive into v2: word level localization via forced alignment, and live fact checking against a retrieved ground truth. The mechanism does not survive, for three reasons.

**Space A trains away the signal it wants.** Contrastive alignment on unlabelled audio and transcript pairs teaches the projection heads that hesitant audio for *"mainframe"* still maps to the token *"mainframe."* That is what a correctly trained aligner does. After convergence, cosine distance encodes "does this audio contain this word," not "did the speaker sound unsure." Prosody is nuisance variance to that training objective. The draft also left the positive and negative pair construction, the loss, and the threshold calibration unspecified, which is where a confidence score of 0.92 in the sample payload came from nowhere.

**Space B asks cosine distance to do natural language inference.** *"The packet goes to the switch"* and *"so the router receives it directly"* are nearly identical in embedding space: same topic, same vocabulary, same register. Contradiction barely moves a sentence embedding. The predicted "clash" is a near coincidence instead. The correct tool is a cross encoder trained on MNLI, and DeBERTa-v3 is already the standard backbone for exactly that, so the encoder choice was right and only the head was wrong.

**Space C is backwards for its own example.** *"DNS translates IPs to domains"* and *"DNS translates domains to IPs"* share every content word. Bi-encoder similarity is close to order insensitive, so the argument reversal that constitutes the entire error is invisible to the metric. Retrieval will pull the correct passage. Verification then needs entailment, not distance.

The common failure is that cosine distance conflates *about the same thing* with *says the same thing*, and two of the three spaces depend on the second meaning.

### A.2 What was kept

| From the tri-modal draft | Form in v2 |
|---|---|
| Live confusion detection, not post hoc only | Instruments B and C both stream |
| Word and clause level localization | Forced alignment, shared segment and clause IDs (§7) |
| Retrieval against ground truth | C2, with an explicit source document per session (§4.1) |
| Fact and logic errors as distinct failure types | C1 self contradiction, C2 source contradiction |
| DeBERTa-v3 as the text model | Used as an MNLI cross encoder rather than a bi-encoder (§6.2) |
| Frozen encoders, train only a small head | Instrument B tier 2, frozen wav2vec2 with a trained head (§5.3) |
| Structured JSON payload driving the UI | §5.7 and §6.5 |
| A worked inference example in the doc | §9 |

### A.3 The experiment that would overturn this

§11 includes an ablation of retrieval plus NLI against a pure cosine distance baseline on the same hand labelled clause set. Two hours of work. If cosine distance reaches comparable precision on contradiction detection, this appendix is wrong and the simpler design wins. Run it.

---

## Appendix B: Open decisions

Four calls the team needs to make. Written down so they get made once.

1. **Source document requirement.** v2 assumes every session is seeded with a source. This unlocks C2 and better questions but constrains topic freedom. Alternative: allow freeform topics that run C1 only. *Recommendation: curated sources for the demo, freeform as a later mode.*
2. **Default session mode.** Measurement first then tutoring is the assumed flow. Alternative: tutoring always, and accept a contaminated delta in exchange for a better experience. *Recommendation: keep measurement first, it is the whole claim.*
3. **Persona base model diversity.** One base model with seeds is cheap but correlated. Two vendors halves the correlation and doubles the integration work. *Recommendation: one model, report the limitation, split arms only if a second key is already working.*
4. **Answer key coverage.** MCQ and numeric are fully mechanical. Short constrained answers still need a verifier model. *Recommendation: force MCQ or numeric for at least 70% of surviving questions and report the split, so "mechanical" is a fact rather than a slogan.*