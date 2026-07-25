import { useCallback, useEffect, useRef, useState } from "react";
import { downsample, encodeWav, mergeBuffers } from "./useWavRecorder";

/**
 * Hands-free capture for the live classroom. You arm the mic once and teach; every natural pause
 * closes an utterance and hands it back as a 16 kHz mono WAV — the format the ml-service wants.
 *
 * The tuning below is what separates this from a toy VAD. Each constant exists because of a
 * specific way naive silence detection fails on a real laptop mic in a real room.
 */
const SILENCE_HOLD_MS = 3_000;   // the pause that ends an utterance
const MIN_SPEECH_MS = 1_200;     // below this it's a cough or a chair, not a teaching turn
const MAX_UTTERANCE_MS = 45_000; // force-cut a monologue so it still gets analyzed
const PRE_ROLL_MS = 300;         // audio kept from BEFORE speech was detected (see below)
const CALIBRATION_MS = 600;      // how long we listen to the room before trusting the threshold
const FLOOR_MIN = 0.006;         // a silent room still has a noise floor; don't divide into zero
const BLOCK_SIZE = 4096;         // ~85 ms at 48 kHz

/** A pause shorter than SILENCE_HOLD_MS, but long enough to be a stall rather than a word gap. */
const HESITATION_PAUSE_MS = 350;

export type RecorderState = "idle" | "calibrating" | "waiting" | "speaking";

/**
 * How the utterance *sounded*, measured from the same RMS blocks the VAD already runs on.
 *
 * This exists because the GPU's hesitation score is z-scored within an utterance and so cannot
 * see speech that is unsure the whole way through — hedge from start to finish and no word is an
 * outlier. Pauses and dead air are absolute, need no model, and are already being computed here.
 */
export type SpeechProsody = {
  speech_ms: number;
  total_ms: number;
  /** Internal stalls over HESITATION_PAUSE_MS that did NOT end the utterance. */
  pause_count: number;
  longest_pause_ms: number;
  mean_level: number;
  peak_level: number;
};

type Options = {
  onUtterance: (audio: Blob, prosody: SpeechProsody) => void;
};

export function useContinuousRecorder({ onUtterance }: Options) {
  const [state, setState] = useState<RecorderState>("idle");
  const [level, setLevel] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const contextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);

  // Ring of recent blocks kept while nobody is speaking. When speech starts we prepend these, so
  // the chunk doesn't begin mid-word — without it every utterance loses its first consonant and
  // the transcript comes back as gibberish.
  const preRollRef = useRef<Float32Array[]>([]);
  const speechRef = useRef<Float32Array[]>([]);
  const sampleRateRef = useRef(48_000);
  const thresholdRef = useRef(FLOOR_MIN);
  const floorSamplesRef = useRef<number[]>([]);
  const calibratedAtRef = useRef(0);
  const speechStartedAtRef = useRef(0);
  const lastVoiceAtRef = useRef(0);
  const isSpeakingRef = useRef(false);

  // Prosody accumulators for the utterance currently open.
  const voicedBlocksRef = useRef(0);
  const totalBlocksRef = useRef(0);
  const levelSumRef = useRef(0);
  const peakLevelRef = useRef(0);
  const pauseCountRef = useRef(0);
  const longestPauseRef = useRef(0);
  const inPauseRef = useRef(false);
  const blockMsRef = useRef((BLOCK_SIZE / 48_000) * 1000);
  const onUtteranceRef = useRef(onUtterance);
  onUtteranceRef.current = onUtterance;

  const teardown = useCallback(() => {
    processorRef.current?.disconnect();
    sourceRef.current?.disconnect();
    streamRef.current?.getTracks().forEach((track) => track.stop());
    void contextRef.current?.close();
    processorRef.current = null;
    sourceRef.current = null;
    streamRef.current = null;
    contextRef.current = null;
    preRollRef.current = [];
    speechRef.current = [];
    isSpeakingRef.current = false;
  }, []);

  const resetProsody = useCallback(() => {
    voicedBlocksRef.current = 0;
    totalBlocksRef.current = 0;
    levelSumRef.current = 0;
    peakLevelRef.current = 0;
    pauseCountRef.current = 0;
    longestPauseRef.current = 0;
    inPauseRef.current = false;
  }, []);

  const emit = useCallback((blocks: Float32Array[], blockMs: number) => {
    const merged = mergeBuffers(blocks);
    if (merged.length === 0) return;
    const total = totalBlocksRef.current || 1;
    const prosody: SpeechProsody = {
      speech_ms: Math.round(voicedBlocksRef.current * blockMs),
      total_ms: Math.round(totalBlocksRef.current * blockMs),
      pause_count: pauseCountRef.current,
      longest_pause_ms: Math.round(longestPauseRef.current),
      mean_level: Number((levelSumRef.current / total).toFixed(5)),
      peak_level: Number(peakLevelRef.current.toFixed(5)),
    };
    onUtteranceRef.current(encodeWav(downsample(merged, sampleRateRef.current)), prosody);
    resetProsody();
  }, [resetProsody]);

  const arm = useCallback(async () => {
    if (contextRef.current) return;
    setError(null);
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        // Let the browser clean up the room for us; the VAD only has to beat what's left.
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
    } catch {
      setError("MICROPHONE ACCESS WAS DENIED OR IS UNAVAILABLE.");
      return;
    }

    const context = new AudioContext();
    const source = context.createMediaStreamSource(stream);
    const processor = context.createScriptProcessor(BLOCK_SIZE, 1, 1);

    sampleRateRef.current = context.sampleRate;
    preRollRef.current = [];
    speechRef.current = [];
    floorSamplesRef.current = [];
    thresholdRef.current = FLOOR_MIN;
    calibratedAtRef.current = 0;
    isSpeakingRef.current = false;

    const blockMs = (BLOCK_SIZE / context.sampleRate) * 1000;
    blockMsRef.current = blockMs;
    const preRollBlocks = Math.max(1, Math.ceil(PRE_ROLL_MS / blockMs));
    const startedAt = performance.now();

    processor.onaudioprocess = (event) => {
      const block = new Float32Array(event.inputBuffer.getChannelData(0));
      let sum = 0;
      for (let i = 0; i < block.length; i += 1) sum += block[i] * block[i];
      const rms = Math.sqrt(sum / block.length);
      const now = performance.now();
      setLevel(rms);

      // Phase 1 — learn the room. A fixed threshold is the classic failure: a laptop fan or an
      // air conditioner sits above it and the VAD never closes an utterance, or the room is dead
      // quiet and it never opens one.
      if (now - startedAt < CALIBRATION_MS) {
        floorSamplesRef.current.push(rms);
        return;
      }
      if (calibratedAtRef.current === 0) {
        const samples = floorSamplesRef.current;
        const floor = samples.length
          ? samples.slice().sort((a, b) => a - b)[Math.floor(samples.length / 2)]
          : 0;
        thresholdRef.current = Math.max(floor * 3, FLOOR_MIN);
        calibratedAtRef.current = now;
        lastVoiceAtRef.current = now;
        setState("waiting");
      }

      const isVoice = rms > thresholdRef.current;

      if (!isSpeakingRef.current) {
        preRollRef.current.push(block);
        if (preRollRef.current.length > preRollBlocks) preRollRef.current.shift();
        if (!isVoice) return;

        isSpeakingRef.current = true;
        speechStartedAtRef.current = now;
        lastVoiceAtRef.current = now;
        speechRef.current = [...preRollRef.current];   // keep the run-up
        preRollRef.current = [];
        resetProsody();
        setState("speaking");
        return;
      }

      speechRef.current.push(block);

      // --- prosody for the open utterance ---
      totalBlocksRef.current += 1;
      levelSumRef.current += rms;
      if (rms > peakLevelRef.current) peakLevelRef.current = rms;
      if (isVoice) {
        voicedBlocksRef.current += 1;
        // A stall that has now ended: count it once, on the way out.
        if (inPauseRef.current) {
          const pauseMs = now - lastVoiceAtRef.current;
          if (pauseMs >= HESITATION_PAUSE_MS) {
            pauseCountRef.current += 1;
            longestPauseRef.current = Math.max(longestPauseRef.current, pauseMs);
          }
          inPauseRef.current = false;
        }
        lastVoiceAtRef.current = now;
      } else if (now - lastVoiceAtRef.current >= HESITATION_PAUSE_MS) {
        inPauseRef.current = true;
      }

      const speechMs = now - speechStartedAtRef.current;
      const quietMs = now - lastVoiceAtRef.current;

      if (quietMs < SILENCE_HOLD_MS && speechMs < MAX_UTTERANCE_MS) return;

      // Close the utterance. Anything shorter than MIN_SPEECH_MS of actual voice is discarded
      // rather than sent — a door closing should not become a teaching turn with an LLM bill.
      const voicedMs = speechMs - quietMs;
      const blocks = speechRef.current;
      isSpeakingRef.current = false;
      speechRef.current = [];
      preRollRef.current = [];
      setState("waiting");
      if (voicedMs >= MIN_SPEECH_MS) emit(blocks, blockMs);
    };

    source.connect(processor);
    processor.connect(context.destination);
    contextRef.current = context;
    streamRef.current = stream;
    sourceRef.current = source;
    processorRef.current = processor;
    setState("calibrating");
  }, [emit, resetProsody]);

  const disarm = useCallback(() => {
    // Don't discard a sentence in progress just because the teacher clicked stop mid-thought.
    if (isSpeakingRef.current && speechRef.current.length > 0) {
      const speechMs = performance.now() - speechStartedAtRef.current;
      if (speechMs >= MIN_SPEECH_MS) emit(speechRef.current, blockMsRef.current);
    }
    teardown();
    setState("idle");
    setLevel(0);
  }, [emit, teardown]);

  useEffect(() => teardown, [teardown]);

  return {
    state,
    level,
    error,
    threshold: thresholdRef.current,
    isArmed: state !== "idle",
    isSpeaking: state === "speaking",
    arm,
    disarm,
  };
}
