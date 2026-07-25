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

/** The meter saturates around a normal speaking level so quiet speech still moves it visibly. */
const METER_GAIN = 900;

export type RecorderState = "idle" | "calibrating" | "waiting" | "speaking";

/** Attach to the `.mic-meter-fill` div; the recorder drives that node's width itself. */
export type MeterRef = (node: HTMLDivElement | null) => void;

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
  const [error, setError] = useState<string | null>(null);

  const contextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);

  // The meter is decoration that moves ~11x/second. Through useState that re-rendered the whole
  // class on every audio block, which starved effects owning timers — Classroom's 520 ms zoom was
  // cleared and re-armed faster than it could fire, so clicking a student never opened the zoom.
  // The value lives in a ref and a rAF loop writes it onto the node the consumer hands us: smoother
  // than the block rate, and zero renders.
  const levelRef = useRef(0);
  const meterNodeRef = useRef<HTMLDivElement | null>(null);
  const meterFrameRef = useRef(0);
  const paintedPctRef = useRef(-1);

  // Every arm() stamps itself with a token before it awaits getUserMedia. teardown() bumps it, so
  // an arm that resolves after a disarm/unmount can tell the stream in its hand is unwanted — it is
  // the only code still holding that stream, so if it doesn't stop it nothing will and the browser
  // keeps the recording indicator lit for the life of the tab.
  const armTokenRef = useRef(0);

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

  const paintMeter = useCallback(() => {
    const node = meterNodeRef.current;
    if (!node) return;
    const pct = Math.min(100, Math.round(levelRef.current * METER_GAIN));
    // Skipping an unchanged value stops the CSS step-transition restarting 60x/second.
    if (pct === paintedPctRef.current) return;
    paintedPctRef.current = pct;
    node.style.width = `${pct}%`;
  }, []);

  // Painting on attach matters: refs run during commit, before the browser paints, so a bar that
  // mounts mid-session never flashes at its CSS default width.
  const meterRef = useCallback<MeterRef>((node) => {
    meterNodeRef.current = node;
    paintedPctRef.current = -1;
    paintMeter();
  }, [paintMeter]);

  // Reads the node every frame rather than capturing it: Classroom only mounts the bar once armed
  // (after this starts), while BackendTeachingWorkspace mounts it before arm() resolves. Both work.
  const startMeterLoop = useCallback(() => {
    if (meterFrameRef.current) return;
    const frame = () => {
      meterFrameRef.current = requestAnimationFrame(frame);
      paintMeter();
    };
    meterFrameRef.current = requestAnimationFrame(frame);
  }, [paintMeter]);

  const teardown = useCallback(() => {
    // Bump first. An arm() parked on getUserMedia re-reads this the moment it resumes; without the
    // bump it goes on to build a live AudioContext + MediaStream that no ref points at.
    armTokenRef.current += 1;
    if (meterFrameRef.current) cancelAnimationFrame(meterFrameRef.current);
    meterFrameRef.current = 0;
    levelRef.current = 0;
    paintedPctRef.current = -1;
    // Written inline rather than via paintMeter() to keep this callback's deps empty — see the
    // unmount effect at the bottom, which makes teardown's identity the mic's kill switch.
    if (meterNodeRef.current) meterNodeRef.current.style.width = "0%";
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
    // Claim a token AFTER that guard and BEFORE the await. The guard only rejects an arm once a
    // previous one has finished; two arms overlapping inside the await both sail past it (React
    // StrictMode's mount/unmount/mount does exactly that), so the loser needs a way to know it lost.
    const token = (armTokenRef.current += 1);
    setError(null);
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        // Let the browser clean up the room for us; the VAD only has to beat what's left.
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
    } catch {
      // Superseded arms stay quiet: a denial from a session the teacher already left must not paint
      // an error over the one that is now running.
      if (armTokenRef.current === token) setError("MICROPHONE ACCESS WAS DENIED OR IS UNAVAILABLE.");
      return;
    }

    // Disarmed, unmounted, or overtaken while the permission prompt was open. Everything below is
    // synchronous, so this one check covers the whole race window.
    if (armTokenRef.current !== token) {
      stream.getTracks().forEach((track) => track.stop());
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
      levelRef.current = rms;

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
    startMeterLoop();
    setState("calibrating");
  }, [emit, resetProsody, startMeterLoop]);

  const disarm = useCallback(() => {
    // Don't discard a sentence in progress just because the teacher clicked stop mid-thought.
    if (isSpeakingRef.current && speechRef.current.length > 0) {
      const speechMs = performance.now() - speechStartedAtRef.current;
      if (speechMs >= MIN_SPEECH_MS) emit(speechRef.current, blockMsRef.current);
    }
    teardown();          // also zeroes the meter, so a bar still on screen lands at empty
    setState("idle");
  }, [emit, teardown]);

  // teardown's deps are empty on purpose: if its identity churned, this cleanup would fire and cut
  // the mic mid-class on an unrelated re-render.
  useEffect(() => teardown, [teardown]);

  return {
    state,
    error,
    meterRef,
    threshold: thresholdRef.current,
    isArmed: state !== "idle",
    isSpeaking: state === "speaking",
    arm,
    disarm,
  };
}
