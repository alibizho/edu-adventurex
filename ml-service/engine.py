"""ConfusionEngine — the tri-modal detector, refactored from audi.ipynb for hosting.

Changes vs the notebook:
  - one class, models loaded once (not at import); no module-global rolling context — history is
    passed per request so the service is stateless and multi-session safe.
  - Whisper large-v3 -> large-v3-turbo/int8; encoders in fp16; torch.inference_mode everywhere.
  - no per-request temp WAV: numpy audio goes straight into faster-whisper.
  - judge is pluggable: local 4-bit Qwen (~1 GB) or an OpenAI-compatible API (0 GB on the box).
  - Space C is LLM-only fact-check (no BGE/Chroma/DB) — the judge decides from its own knowledge.
    Toggle with ENABLE_SPACE_C. Swap back to RAG for syllabus-grounded/citable checking.
  - output is the backend ChunkAnalysis contract, with internal 'dissonance' mapped to a [0,1]
    confidence (HIGH = clear).

Localization fixes (notebook review):
  1. Space A pools sub-token distances into words via offset_mapping (handles MAC->[M, AC]),
     instead of the proportional word/token guess.
  2. Space B/C judge returns the exact offending span (+ correction); we locate THAT word,
     not the last / hesitation word.

Extra signals folded in (cognitive-load prototype):
  - pace: seconds-per-char per word (from Whisper timestamps), z-scored -> 'bottleneck' when a
    word took abnormally long to say.
  - attention entropy: per-token entropy of the cross-attention -> 'scattered' alignment when the
    model can't localize a word to specific audio frames. Both pooled to words the same way.

Curriculum grounding (added once the backend had a teaching plan to send):
  - the caller passes the class objective/notes/material and the concepts already covered, so
    Space C judges against THAT rather than the judge's general knowledge, and can distinguish
    wrong (CONTRADICTION) from off-syllabus (OFF_TOPIC) from correct-but-advanced (BEYOND).
  - cross-modal fusion: when the words are right but the delivery is laboured, that's a
    fluency_issue — recitation without understanding, which no single space catches alone.
  - the AI student's question is written here, from the strongest anomaly, while the word-level
    evidence is still in hand; the backend just relays it.
"""
import os
import re

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

import config as C
from alignment import AlignmentEngine
from schemas import (Anomaly, BEYOND, ChunkAnalysis, COGNITIVE_LOAD, CurriculumUpdate,
                     FACTUAL_ERROR, FLUENCY_ISSUE, LOGIC_ERROR, OFF_TOPIC, RECALL_FAILURE,
                     StudentQuestion, WordScore)


class ConfusionEngine:
    def __init__(self) -> None:
        self.device = C.DEVICE
        print(f"[engine] device={self.device}  whisper={C.WHISPER_MODEL}/{C.WHISPER_COMPUTE}  "
              f"space_c={C.ENABLE_SPACE_C}  judge={C.JUDGE_BACKEND}")
        self._load_asr()
        self._load_encoders()
        self._load_brain()
        self._load_judge()
        print("[engine] ready.\n")

    # ---------- model loading ----------

    def _load_asr(self) -> None:
        from faster_whisper import WhisperModel
        self.whisper = WhisperModel(C.WHISPER_MODEL, device=self.device, compute_type=C.WHISPER_COMPUTE)

    def _load_encoders(self) -> None:
        from transformers import (AutoModel, AutoTokenizer, Wav2Vec2FeatureExtractor,
                                   Wav2Vec2Model)
        dt = torch.float16 if self.device == "cuda" else torch.float32
        w2v_local, deb_local = os.path.isdir(C.WAV2VEC_SRC), os.path.isdir(C.DEBERTA_SRC)
        self.audio_extractor = Wav2Vec2FeatureExtractor.from_pretrained(C.WAV2VEC_SRC, local_files_only=w2v_local)
        self.audio_model = Wav2Vec2Model.from_pretrained(C.WAV2VEC_SRC, local_files_only=w2v_local).to(self.device, dt).eval()
        self.text_tokenizer = AutoTokenizer.from_pretrained(C.DEBERTA_SRC, local_files_only=deb_local)
        self.text_model = AutoModel.from_pretrained(C.DEBERTA_SRC, local_files_only=deb_local).to(self.device, dt).eval()
        self._enc_dtype = dt

    def _load_brain(self) -> None:
        self.brain = AlignmentEngine().to(self.device).eval()
        path = next((p for p in C.ALIGN_WEIGHTS if p and os.path.exists(p)), None)
        if path is None:
            raise FileNotFoundError(f"No alignment weights found in {C.ALIGN_WEIGHTS}")

        raw = torch.load(path, map_location=self.device)
        sd = raw
        if isinstance(raw, dict):
            for wrap in ("state_dict", "model_state_dict", "model"):
                if isinstance(raw.get(wrap), dict):
                    sd = raw[wrap]
                    break

        # The training model wraps AlignmentEngine as a submodule `alignment.*` and adds a
        # training-only `contrastive_head.*`. Keep the alignment weights (strip the prefix), drop
        # the head — it isn't used at inference.
        inner = {k.split("alignment.", 1)[1]: v for k, v in sd.items() if k.startswith("alignment.")}
        if not inner:  # already an unwrapped AlignmentEngine checkpoint
            inner = {k: v for k, v in sd.items() if not k.startswith("contrastive_head.")}

        missing, unexpected = self.brain.load_state_dict(inner, strict=False)
        if missing:
            raise RuntimeError(f"alignment brain missing tensors after remap: {list(missing)}")
        print(f"[engine] alignment brain: {path}  (loaded {len(inner)} tensors, "
              f"dropped contrastive_head + {len(unexpected)} unexpected)")

    def _load_judge(self) -> None:
        if C.JUDGE_BACKEND == "api":
            from openai import OpenAI
            self._judge_client = OpenAI(base_url=C.JUDGE_API_BASE, api_key=C.JUDGE_API_KEY or "not-set")
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.qwen_tok = AutoTokenizer.from_pretrained(C.QWEN_MODEL, trust_remote_code=True)
        kwargs: dict = {"trust_remote_code": True}
        if C.QWEN_4BIT and self.device == "cuda":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
            kwargs["device_map"] = "auto"
        else:
            kwargs["torch_dtype"] = torch.float16 if self.device == "cuda" else torch.float32
            kwargs["device_map"] = "auto" if self.device == "cuda" else None
        self.qwen = AutoModelForCausalLM.from_pretrained(C.QWEN_MODEL, **kwargs)

    # ---------- judge (Space B / C / question writing) ----------

    def _judge(self, prompt: str, max_tokens: int | None = None, temperature: float = 0.0) -> str:
        """One judge completion. max_tokens defaults to JUDGE_MAX_TOKENS — far above the notebook's
        8 — because the judge now returns a span + correction, and writes the student's question.
        Any failure (bad key, timeout, rate limit) degrades to '' -> parsed as no-contradiction, so
        /analyze never 500s on the judge; the warning tells you to fix it. temperature > 0 is only
        used for question writing, where identical phrasing every turn would be obvious."""
        if max_tokens is None:
            max_tokens = C.JUDGE_MAX_TOKENS
        try:
            if C.JUDGE_BACKEND == "api":
                r = self._judge_client.chat.completions.create(
                    model=C.JUDGE_API_MODEL, temperature=temperature, max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}])
                return (r.choices[0].message.content or "").strip()
            messages = [{"role": "user", "content": prompt}]
            text = self.qwen_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.qwen_tok([text], return_tensors="pt").to(self.qwen.device)
            with torch.inference_mode():
                out = self.qwen.generate(
                    inputs.input_ids, max_new_tokens=max_tokens, do_sample=(temperature > 0),
                    temperature=temperature if temperature > 0 else None)
            return self.qwen_tok.batch_decode(out[:, inputs.input_ids.shape[1]:],
                                              skip_special_tokens=True)[0].strip()
        except Exception as e:
            print(f"[engine] judge call failed ({e!r}); treating as no-contradiction")
            return ""

    @staticmethod
    def _parse_verdict(out: str) -> dict:
        """Parse 'VERDICT | offending span | correction' (span/correction optional).

        Order matters: a BEYOND/OFF_TOPIC reply often also contains the word "aligned" in its
        explanation, so the more specific verdicts are checked first."""
        up = out.upper()
        verdict = "CONSISTENT"
        if "CONTRADICTION" in up:
            verdict = "CONTRADICTION"
        elif "BEYOND" in up:
            verdict = "BEYOND"
        elif "OFF_TOPIC" in up:
            verdict = "OFF_TOPIC"
        elif "ALIGNED" in up:
            verdict = "ALIGNED"

        span = correction = None
        if "|" in out:
            parts = [p.strip() for p in out.split("|")]
            if len(parts) >= 2 and parts[1]:
                span = parts[1]
            if len(parts) >= 3 and parts[2]:
                correction = parts[2]
        return {"verdict": verdict, "span": span, "correction": correction}

    # ---------- main entry ----------

    def analyze(self, audio_path: str, history: list[str], chunk_id: int = 0,
                enable_space_c: bool | None = None, overall_topic: str = "",
                curriculum_context: str = "", key_concepts: list[str] | None = None) -> ChunkAnalysis:
        use_c = C.ENABLE_SPACE_C if enable_space_c is None else enable_space_c
        key_concepts = key_concepts or []

        y, transcript, words, lang = self._transcribe(audio_path)
        if not words:
            return ChunkAnalysis(chunk_id=chunk_id, text="", confidence=1.0)

        sa = self._space_a(y, transcript, words)
        n_w = len(words)
        red = [i for i, z in enumerate(sa["z"]) if z > C.ZSCORE_ANOMALY and 0 < i < n_w - 1]

        logic = self._space_b(transcript, history)
        fact = self._space_c(transcript, curriculum_context, key_concepts) if use_c else {"verdict": "SKIP"}

        return self._build(chunk_id, transcript, words, sa, red, logic, fact, history,
                           overall_topic, curriculum_context)

    # ---------- stages ----------

    def _transcribe(self, audio_path: str):
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        y = waveform.squeeze(0).numpy().astype(np.float32)      # feed numpy straight in, no temp WAV

        # Preferred path: real per-word timestamps (they drive the pace signal). Some turbo CT2
        # builds crash in model.align() with std::bad_alloc — catch it and fall back to segment
        # timing so /analyze never 500s. Space A/B/C don't depend on these timestamps.
        try:
            segments, info = self.whisper.transcribe(y, word_timestamps=True, vad_filter=True)
            words = [{"word": w.word.strip(), "start": w.start, "end": w.end}
                     for seg in segments if seg.words for w in seg.words]
            if words:
                return y, " ".join(w["word"] for w in words), words, info.language
        except Exception as e:
            print(f"[engine] word-timestamp alignment failed ({e!r}); using segment-level timing")

        # Fallback: no word alignment. Spread each segment's [start,end] across its words by char
        # length — approximate pace, but robust.
        segments, info = self.whisper.transcribe(y, word_timestamps=False, vad_filter=True)
        words = []
        for seg in segments:
            toks = seg.text.split()
            if not toks:
                continue
            dur = max(1e-3, seg.end - seg.start)
            total = sum(len(t) for t in toks) or 1
            t = seg.start
            for tok in toks:
                share = dur * len(tok) / total
                words.append({"word": tok, "start": t, "end": t + share})
                t += share
        return y, " ".join(w["word"] for w in words), words, info.language

    def _space_a(self, y, transcript, words) -> dict:
        """Audio vs text: per-WORD hesitation (z), articulation pace (z), and attention entropy."""
        inputs_a = self.audio_extractor(y, sampling_rate=16000, return_tensors="pt").to(self.device)
        enc = self.text_tokenizer(transcript, return_tensors="pt", padding=True, truncation=True,
                                  return_offsets_mapping=self.text_tokenizer.is_fast)
        offsets = enc.pop("offset_mapping")[0].tolist() if self.text_tokenizer.is_fast else None
        inputs_t = {k: v.to(self.device) for k, v in enc.items()}

        with torch.inference_mode():
            a_feats = self.audio_model(inputs_a.input_values.to(self._enc_dtype)).last_hidden_state.float()
            t_feats = self.text_model(**inputs_t).last_hidden_state.float()
            aligned_audio, norm_text, attn = self.brain(a_feats, t_feats)
            tok_dist = 1.0 - F.cosine_similarity(aligned_audio, norm_text, dim=-1).squeeze(0).cpu().numpy()
            attn = attn.squeeze(0).cpu().numpy()               # [n_text_tokens, n_audio_frames]

        # attention entropy per text token: high = alignment couldn't settle on audio frames.
        p = np.clip(attn, 1e-9, 1.0)
        tok_entropy = -(p * np.log(p)).sum(axis=1)

        word_scores = self._pool_to_words(tok_dist, transcript, words, offsets, agg="max")
        entropy_w = self._pool_to_words(tok_entropy, transcript, words, offsets, agg="max")

        mean_ws, std_ws = word_scores.mean(), max(word_scores.std(), 1e-6)
        z = (word_scores - mean_ws) / std_ws

        # cognitive load: articulation time per character, z-scored across the utterance.
        rate = np.array([max(1e-3, w["end"] - w["start"]) / max(len(w["word"].strip()), 1) for w in words])
        pm, ps = rate.mean(), max(rate.std(), 1e-6)
        pace_z = (rate - pm) / ps

        return {"z": z, "raw": word_scores, "pace_z": pace_z, "entropy": entropy_w}

    def _pool_to_words(self, per_token, transcript, words, offsets, agg="max"):
        """Aggregate a per-(sub)token array into per-word values by char-span overlap. 'max' lets
        the worst sub-token drive the word. Proportional fallback only when offsets are missing."""
        num_words = len(words)
        agg_fn = np.max if agg == "max" else np.mean
        if offsets is None:                                    # slow-tokenizer fallback
            num_tokens = len(per_token) - 2
            return np.array([
                float(per_token[int((i / max(num_words, 1)) * num_tokens) + 1])
                if int((i / max(num_words, 1)) * num_tokens) + 1 < len(per_token) else 0.0
                for i in range(num_words)], dtype=float)

        spans, pos = [], 0
        for w in words:
            idx = transcript.find(w["word"], pos)
            if idx < 0:
                idx = pos
            spans.append((idx, idx + len(w["word"])))
            pos = idx + len(w["word"])

        out = []
        for (ws, we) in spans:
            vals = [per_token[i] for i, (s, e) in enumerate(offsets)
                    if e > s and min(we, e) > max(ws, s)]      # sub-tokens overlapping this word
            out.append(float(agg_fn(vals)) if vals else 0.0)
        return np.array(out, dtype=float)

    def _space_b(self, transcript: str, history: list[str]) -> dict:
        if not history:
            return {"verdict": "CONSISTENT", "span": None, "correction": None}
        ctx = " ".join(history)
        out = self._judge(
            f"EARLIER: {ctx}\nNOW: {transcript}\n"
            "Does NOW contradict EARLIER? Reply ONE line:\n"
            "CONSISTENT\n"
            "CONTRADICTION | <exact words>")
        return self._parse_verdict(out)

    def _space_c(self, transcript: str, curriculum_context: str, key_concepts: list[str]) -> dict:
        """Fact check. With curriculum context the judge grades against the material actually being
        taught, which is what makes OFF_TOPIC and BEYOND meaningful — without it there is no
        syllabus to be off, or beyond. Falls back to a bare LLM-knowledge check otherwise."""
        if curriculum_context:
            prompt = (
                f"REFERENCE MATERIAL (Ground Truth):\n{curriculum_context[:1500]}\n\n"
                f"KEY CONCEPTS:\n{', '.join(key_concepts)}\n\n"
                f"STUDENT'S EXPLANATION:\n{transcript}\n\n"
                "Evaluate the explanation against the reference:\n"
                "- ALIGNED: Correct and covers the material.\n"
                "- CONTRADICTION: Factually wrong or contradicts the reference.\n"
                "- BEYOND: Factually correct, but introduces advanced concepts NOT in the reference.\n"
                "- OFF_TOPIC: Completely unrelated to the reference material or key concepts.\n\n"
                "Reply on ONE line:\n"
                "ALIGNED\n"
                "CONTRADICTION | <exact wrong words> | <correction from reference>\n"
                "BEYOND | <the new advanced concept introduced>\n"
                "OFF_TOPIC | <brief reason>"
            )
        else:
            prompt = (
                f"CLAIM: {transcript}\nIs this claim factually correct? Reply on ONE line:\n"
                "ALIGNED\nCONTRADICTION | <exact wrong words> | <correction>"
            )
        out = self._judge(prompt, max_tokens=120)
        return self._parse_verdict(out)

    # ---------- assembly ----------

    def _locate(self, words, span: str | None) -> int:
        """Index of the word that best matches the judge's offending span; -1 if none."""
        if not span:
            return -1
        span_l = span.lower().strip().strip(".,!?;:\"'()[]{}")
        span_toks = {t.strip(".,!?;:\"'()[]{}") for t in span_l.split()}
        for i, w in enumerate(words):
            wl = w["word"].lower().strip(".,!?;:\"'()[]{}")
            if wl and (wl in span_l or wl in span_toks):
                return i
        return -1

    def _build(self, chunk_id, transcript, words, sa, red, logic, fact, history,
               overall_topic, curriculum_context):
        z_scores, pace_z, entropy_w = sa["z"], sa["pace_z"], sa["entropy"]
        n = len(words)

        e_mean, e_std = entropy_w.mean(), max(entropy_w.std(), 1e-6)
        scattered = entropy_w > (e_mean + 2 * e_std)
        bottleneck = pace_z > C.PACE_Z_THRESHOLD

        anomalies: list[Anomaly] = []

        # Space A: hesitation / recall failure — the exact hesitant word.
        a_target = red[0] if red else -1
        if red:
            anomalies.append(Anomaly(
                type=RECALL_FAILURE, source="space_a/audio-text",
                score=round(float(min(1.0, max(z_scores[red]) / 4.0)), 2),
                evidence=f"hesitation on '{words[a_target]['word'].strip()}'"))

        # cognitive load: slow articulation and/or scattered alignment.
        load_idx = [i for i in range(n) if bottleneck[i] or scattered[i]]
        if load_idx:
            worst = max(load_idx, key=lambda i: pace_z[i])
            anomalies.append(Anomaly(
                type=COGNITIVE_LOAD, source="space_a/pace+entropy",
                score=round(float(min(1.0, max(pace_z[load_idx]) / 3.0)), 2),
                evidence=f"slow/scattered delivery on '{words[worst]['word'].strip()}'"))

        # Space B: logic error — the judge's offending span, not the last word.
        b_target = self._locate(words, logic.get("span"))
        if logic.get("verdict") == "CONTRADICTION":
            where = words[b_target]["word"].strip() if b_target >= 0 else (logic.get("span") or "?")
            anomalies.append(Anomaly(
                type=LOGIC_ERROR, source="space_b/text-text", score=0.7,
                evidence=f"'{where}' contradicts earlier: '{' '.join(history)[:80]}'"))

        # Space C: wrong / off-syllabus / past-the-syllabus, against the curriculum context.
        c_target = self._locate(words, fact.get("span"))
        if fact.get("verdict") == "CONTRADICTION":
            where = words[c_target]["word"].strip() if c_target >= 0 else (fact.get("span") or "?")
            corr = fact.get("correction")
            anomalies.append(Anomaly(
                type=FACTUAL_ERROR, source="space_c/text-knowledge", score=0.8,
                evidence=f"'{where}' is wrong" + (f" — correct: {corr}" if corr else "")))
        elif fact.get("verdict") == "OFF_TOPIC":
            anomalies.append(Anomaly(
                type=OFF_TOPIC, source="space_c/text-knowledge", score=0.9,
                evidence=f"Drifted from core material: {fact.get('span') or 'unrelated'}"))
        elif fact.get("verdict") == "BEYOND":
            anomalies.append(Anomaly(
                type=BEYOND, source="space_c/text-knowledge", score=1.0,
                evidence=f"Introduced advanced concept: {fact.get('span')}"))

        # Cross-modal fusion — hollow recitation. The content checks out, but a large share of the
        # words cost visible effort to produce: memorised, not understood. Neither space sees this
        # alone; it only exists in the disagreement between them.
        if (fact.get("verdict") in ("ALIGNED", "BEYOND", "CONSISTENT") and load_idx
                and (len(load_idx) / max(n, 1) > C.FLUENCY_LOAD_RATIO_THRESHOLD)):
            worst_load = max(load_idx, key=lambda i: pace_z[i] + z_scores[i])
            anomalies.append(Anomaly(
                type=FLUENCY_ISSUE, source="space_a+c/cross_modal", score=0.75,
                evidence=f"Said '{words[worst_load]['word'].strip()}' correctly, but delivery was "
                         "highly unconfident/strained."))

        # Primary target, most-diagnostic first: a wrong fact beats drift, beats self-contradiction,
        # beats strained delivery, beats a bare hesitation.
        target_idx = -1
        if fact.get("verdict") == "CONTRADICTION" and c_target >= 0:
            target_idx = c_target
        elif fact.get("verdict") == "OFF_TOPIC":
            target_idx = load_idx[0] if load_idx else 0
        elif logic.get("verdict") == "CONTRADICTION" and b_target >= 0:
            target_idx = b_target
        elif any(a.type == FLUENCY_ISSUE for a in anomalies) and load_idx:
            target_idx = max(load_idx, key=lambda i: pace_z[i] + z_scores[i])
        elif a_target >= 0:
            target_idx = a_target

        localized_target = None
        if target_idx >= 0:
            localized_target = words[target_idx]["word"].strip(".,!?;:\"'()[]{}")
        elif fact.get("verdict") in ("CONTRADICTION", "BEYOND", "OFF_TOPIC") and fact.get("span"):
            localized_target = fact["span"]
        elif logic.get("verdict") == "CONTRADICTION" and logic.get("span"):
            localized_target = logic["span"]

        # dissonance -> confidence in [0,1] (HIGH = clear). Absolute severity, not ratios: one
        # badly-stumbled word in a long sentence still means the speaker lost the thread there.
        conf = 1.0
        if logic.get("verdict") == "CONTRADICTION":
            conf -= 0.3
        if fact.get("verdict") == "CONTRADICTION":
            conf -= 0.4
        if fact.get("verdict") == "OFF_TOPIC":
            conf -= 0.5
        if load_idx:
            conf -= min(0.4, max(pace_z[load_idx]) / 4.0)
        if red:
            conf -= min(0.2, max(z_scores[red]) / 5.0)
        confidence = round(float(min(1.0, max(0.05, conf))), 2)

        # BEYOND is not a failure: the learner went past the syllabus and got it right, so the
        # concept is handed back for the backend to fold into the class instead of being discarded.
        curriculum_update = None
        if fact.get("verdict") == "BEYOND" and fact.get("span"):
            curriculum_update = CurriculumUpdate(added_concepts=[fact["span"]])

        # The AI student's interruption, written here while the word-level evidence is in hand.
        student_q = None
        if anomalies and localized_target:
            primary = next((a for a in anomalies if a.type in (
                FACTUAL_ERROR, OFF_TOPIC, FLUENCY_ISSUE, BEYOND, LOGIC_ERROR, RECALL_FAILURE)),
                anomalies[0])
            student_q = self._generate_student_question(
                primary, localized_target, overall_topic, curriculum_context, transcript)

        red_set = set(red)
        detail = [WordScore(
            word=w["word"],
            hesitation_zscore=round(float(z_scores[i]), 3),
            is_anomaly=(i in red_set),   # matches the emitted anomaly (boundary words excluded)
            pace_zscore=round(float(pace_z[i]), 3),
            is_bottleneck=bool(bottleneck[i]),
            attention_entropy=round(float(entropy_w[i]), 3),
            is_scattered=bool(scattered[i]),
        ) for i, w in enumerate(words)]

        return ChunkAnalysis(chunk_id=chunk_id, text=transcript, confidence=confidence,
                             anomalies=anomalies, localized_target=localized_target, detail=detail,
                             student_question=student_q, curriculum_update=curriculum_update)

    def _generate_student_question(self, anomaly: Anomaly, target: str, overall_topic: str,
                                   curriculum: str, transcript: str) -> StudentQuestion | None:
        """Turn the strongest anomaly into something a classmate would actually say. The anomaly
        type picks the stance: confused about an error, intrigued by a BEYOND, gently redirecting
        an OFF_TOPIC. Returns None when the judge is unreachable or writes nothing usable, and the
        backend then falls back to its own question generator."""
        # The curriculum excerpt keeps the question inside what this class actually covers — without
        # it the judge tends to ask something reasonable but off-syllabus.
        reference = f'\nWhat the class covers: "{curriculum[:600]}"' if curriculum else ""
        prompt = f"""You are a student in a classroom. The teacher just spoke.
Teacher's words: "{transcript}"
Main Topic: "{overall_topic}"{reference}
Anomaly detected: {anomaly.type}
Target/Evidence: {anomaly.evidence}

Rules for your question (max 30 words):
- If OFF_TOPIC: Ask a question that gently bridges their comment back to the main topic ({overall_topic}).
- If fluency_issue: Say "You mentioned {target}, but sounded unsure. Can you explain what that means in your own words?"
- If beyond: Act intrigued. "Wow, how does {target} connect to what we were just learning?"
- If factual_error/recall_failure: Act confused. "Wait, I thought {target} was different? Can you clarify?"

CRITICAL: Output ONLY the question itself. No thinking, no quotes, no explanations. Just the question."""

        raw_q = self._judge(prompt, max_tokens=150, temperature=0.6)
        clean_q = self._clean_question(raw_q)
        if clean_q:
            return StudentQuestion(question_text=clean_q, target_concept=target,
                                   anomaly_type=anomaly.type)
        return None

    @staticmethod
    def _clean_question(text: str) -> str:
        """Salvage the question from a chatty judge. Small instruct models ignore "output only the
        question" often enough that this is load-bearing: a reasoning preamble reaching the UI
        would break the illusion of a classmate speaking."""
        if not text:
            return ""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL).strip()

        # Reasoning-preamble openers: keep the first sentence that is actually a question.
        prefixes = ["We need to", "Let's", "I need to", "The rules", "First,", "Since", "Okay,",
                    "As an AI", "Based on"]
        for p in prefixes:
            if text.startswith(p):
                sentences = re.split(r'(?<=[.!?])\s+', text)
                for s in sentences:
                    if '?' in s:
                        return s.strip().strip('"\'')
                return sentences[-1].strip().strip('"\'')

        sentences = re.split(r'(?<=[.!?])\s+', text)
        for s in sentences:
            if '?' in s:
                return s.strip().strip('"\'')

        return text.strip().strip('"\'')
