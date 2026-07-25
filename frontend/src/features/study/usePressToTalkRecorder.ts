import { useCallback, useEffect, useRef, useState } from "react";
import { downsample, encodeWav, mergeBuffers } from "./useWavRecorder";

/**
 * Press-to-talk capture. One press opens the mic, the next press closes it and hands the whole
 * recording back as a 16 kHz mono WAV — the format the ml-service wants.
 *
 * This replaced a hands-free VAD that armed once and cut an utterance at every 3-second pause.
 * The pause is not a reliable end-of-thought: teachers stop to breathe, to read their notes, to
 * think — and the room answered every one of those as if the sentence were finished, while a fast
 * talker was never interrupted at all. Where a clip ends is now the teacher's decision and nothing
 * else's, which also means the mic is never live without someone having asked for it.
 */
const MAX_CAPTURE_MS = 5 * 60_000;  // safety stop; the backend rejects an audio chunk over 15 MB
const MIN_CAPTURE_MS = 800;         // below this it's a double-click, not a teaching turn
const FLOOR_MIN = 0.006;            // a silent room still has a noise floor; don't divide into zero
const BLOCK_SIZE = 4096;            // ~85 ms at 48 kHz

/** A gap long enough to be a stall rather than the space between two words. */
const HESITATION_PAUSE_MS = 350;

/** The meter saturates around a normal speaking level so quiet speech still moves it visibly. */
const METER_GAIN = 900;

export type RecorderState = "idle" | "starting" | "recording";

/** Attach to the `.mic-meter-fill` div; the recorder drives that node's width itself. */
export type MeterRef = (node: HTMLDivElement | null) => void;

/**
 * How the utterance *sounded*, measured from the RMS of each captured block.
 *
 * This exists because the GPU's hesitation score is z-scored within an utterance and so cannot
 * see speech that is unsure the whole way through — hedge from start to finish and no word is an
 * outlier. Pauses and dead air are absolute and need no model.
 */
export type SpeechProsody = {
  speech_ms: number;
  total_ms: number;
  /** Stalls over HESITATION_PAUSE_MS *between* words, not the silence at either end. */
  pause_count: number;
  longest_pause_ms: number;
  mean_level: number;
  peak_level: number;
};

/**
 * Turn the per-block levels of one recording into prosody.
 *
 * The speaking threshold is derived from the recording itself rather than from a calibration
 * window at the start: a press-to-talk recording has no quiet run-in to calibrate against, since
 * the teacher presses and talks, and listening to the first 600 ms would just measure their
 * opening word.
 *
 * Everything is measured over the voiced span (first voiced block to last), not the whole clip.
 * The seconds between pressing record and starting to speak — and between finishing and pressing
 * again — are the interface, not hesitation, and counting them as dead air would make every
 * unhurried teacher read as unsure.
 */
export function measureProsody(levels: readonly number[], blockMs: number): SpeechProsody {
  const round = (value: number) => Math.round(value);
  if (levels.length === 0) {
    return { speech_ms: 0, total_ms: 0, pause_count: 0, longest_pause_ms: 0, mean_level: 0, peak_level: 0 };
  }

  const sorted = [...levels].sort((a, b) => a - b);
  const loudest = sorted[sorted.length - 1];
  // The quietest twentieth of the clip is the room; even unbroken speech dips between words.
  const room = sorted[Math.floor(sorted.length * 0.05)];
  // The ceiling is what keeps that honest. A teacher who talks from one press to the next may
  // have no real silence in the clip at all, and a gate read off their own voice would score the
  // most fluent recordings as pure dead air — the one reading this must never produce.
  const threshold = Math.min(Math.max(room * 3, FLOOR_MIN), loudest * 0.35);

  const first = levels.findIndex((level) => level > threshold);
  const last = levels.length - 1 - [...levels].reverse().findIndex((level) => level > threshold);
  // No block cleared the threshold: nothing to trim to, so report the clip as it is.
  const span = first === -1 ? levels : levels.slice(first, last + 1);

  let pauseCount = 0;
  let longestPause = 0;
  let quietRun = 0;
  let voiced = 0;
  let total = 0;
  let peak = 0;
  span.forEach((level) => {
    total += level;
    if (level > peak) peak = level;
    if (level > threshold) {
      voiced += 1;
      if (quietRun * blockMs >= HESITATION_PAUSE_MS) {
        pauseCount += 1;
        longestPause = Math.max(longestPause, quietRun * blockMs);
      }
      quietRun = 0;
    } else {
      quietRun += 1;
    }
  });

  return {
    speech_ms: round(voiced * blockMs),
    total_ms: round(span.length * blockMs),
    pause_count: pauseCount,
    longest_pause_ms: round(longestPause),
    mean_level: Number((total / span.length).toFixed(5)),
    peak_level: Number(peak.toFixed(5)),
  };
}

/** `0:07`, `1:42`. The mic caption is the only clock a teacher has while recording. */
export function formatElapsed(seconds: number) {
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
}

type Options = {
  onUtterance: (audio: Blob, prosody: SpeechProsody) => void;
};

export function usePressToTalkRecorder({ onUtterance }: Options) {
  const [state, setState] = useState<RecorderState>("idle");
  const [error, setError] = useState<string | null>(null);
  // Whole seconds only, so a five-minute recording re-renders 300 times rather than 3,000.
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  const contextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);

  // The meter moves ~11x/second. Through useState that re-rendered the whole class on every audio
  // block, which starved effects owning timers — Classroom's 520 ms zoom was cleared and re-armed
  // faster than it could fire, so clicking a student never opened the zoom. The value lives in a
  // ref and a rAF loop writes it onto the node the consumer hands us: zero renders.
  const levelRef = useRef(0);
  const meterNodeRef = useRef<HTMLDivElement | null>(null);
  const meterFrameRef = useRef(0);
  const paintedPctRef = useRef(-1);

  // Every start() stamps itself with a token before it awaits getUserMedia. teardown() bumps it, so
  // a start that resolves after a stop/unmount can tell the stream in its hand is unwanted — it is
  // the only code still holding that stream, so if it doesn't stop it nothing will and the browser
  // keeps the recording indicator lit for the life of the tab.
  const startTokenRef = useRef(0);

  const blocksRef = useRef<Float32Array[]>([]);
  const levelsRef = useRef<number[]>([]);
  const sampleRateRef = useRef(48_000);
  const blockMsRef = useRef((BLOCK_SIZE / 48_000) * 1000);
  const onUtteranceRef = useRef(onUtterance);
  onUtteranceRef.current = onUtterance;

  // The 5-minute stop fires from inside onaudioprocess, which is outside React's world and holds
  // whatever closure it was created with. It calls through this ref so it always reaches the
  // current stop() instead of the one that existed when the recording started.
  const stopRef = useRef<() => void>(() => {});

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
  // mounts mid-recording never flashes at its CSS default width.
  const meterRef = useCallback<MeterRef>((node) => {
    meterNodeRef.current = node;
    paintedPctRef.current = -1;
    paintMeter();
  }, [paintMeter]);

  const startMeterLoop = useCallback(() => {
    if (meterFrameRef.current) return;
    const frame = () => {
      meterFrameRef.current = requestAnimationFrame(frame);
      paintMeter();
    };
    meterFrameRef.current = requestAnimationFrame(frame);
  }, [paintMeter]);

  const teardown = useCallback(() => {
    // Bump first. A start() parked on getUserMedia re-reads this the moment it resumes; without the
    // bump it goes on to build a live AudioContext + MediaStream that no ref points at.
    startTokenRef.current += 1;
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
  }, []);

  const start = useCallback(async () => {
    if (contextRef.current) return;
    // Claim a token AFTER that guard and BEFORE the await. The guard only rejects a start once a
    // previous one has finished; two starts overlapping inside the await both sail past it (React
    // StrictMode's mount/unmount/mount does exactly that), so the loser needs a way to know it lost.
    const token = (startTokenRef.current += 1);
    setError(null);
    setElapsedSeconds(0);
    setState("starting");
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        // Let the browser clean up the room for us.
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
    } catch {
      // Superseded starts stay quiet: a denial from a session the teacher already left must not
      // paint an error over the one that is now running.
      if (startTokenRef.current === token) {
        setError("MICROPHONE ACCESS WAS DENIED OR IS UNAVAILABLE.");
        setState("idle");
      }
      return;
    }

    // Stopped, unmounted, or overtaken while the permission prompt was open. Everything below is
    // synchronous, so this one check covers the whole race window.
    if (startTokenRef.current !== token) {
      stream.getTracks().forEach((track) => track.stop());
      return;
    }

    const context = new AudioContext();
    const source = context.createMediaStreamSource(stream);
    const processor = context.createScriptProcessor(BLOCK_SIZE, 1, 1);

    sampleRateRef.current = context.sampleRate;
    blocksRef.current = [];
    levelsRef.current = [];
    const blockMs = (BLOCK_SIZE / context.sampleRate) * 1000;
    blockMsRef.current = blockMs;
    const startedAt = performance.now();

    processor.onaudioprocess = (event) => {
      const block = new Float32Array(event.inputBuffer.getChannelData(0));
      let sum = 0;
      for (let i = 0; i < block.length; i += 1) sum += block[i] * block[i];
      const rms = Math.sqrt(sum / block.length);
      levelRef.current = rms;
      blocksRef.current.push(block);
      levelsRef.current.push(rms);
      // Nobody meant to record for five minutes; this is the guard against a mic left on by
      // accident, and it sends what was captured rather than throwing it away. Said out loud,
      // because a mic that closed on its own is exactly what this whole hook exists to avoid.
      if (performance.now() - startedAt >= MAX_CAPTURE_MS) {
        stopRef.current();
        setError("FIVE MINUTES IS THE LIMIT — THAT CLIP WAS SENT. PRESS THE MIC TO CARRY ON.");
      }
    };

    source.connect(processor);
    processor.connect(context.destination);
    contextRef.current = context;
    streamRef.current = stream;
    sourceRef.current = source;
    processorRef.current = processor;
    startMeterLoop();
    setState("recording");
  }, [startMeterLoop]);

  const stop = useCallback(() => {
    if (!contextRef.current) {
      // Nothing is open: it never started, the 5-minute guard already closed it, or the teacher
      // pressed again while the permission prompt was still up. teardown() covers that last case —
      // it bumps the token, so the start() parked on getUserMedia drops the stream it receives
      // instead of quietly opening a mic nobody is watching any more.
      teardown();
      setState("idle");
      return;
    }
    const blocks = blocksRef.current;
    const levels = levelsRef.current;
    const blockMs = blockMsRef.current;
    blocksRef.current = [];
    levelsRef.current = [];
    teardown();
    setState("idle");
    setElapsedSeconds(0);

    const capturedMs = levels.length * blockMs;
    if (capturedMs < MIN_CAPTURE_MS) {
      setError("THAT WAS TOO SHORT — KEEP THE MIC ON WHILE YOU TEACH.");
      return;
    }
    const prosody = measureProsody(levels, blockMs);
    // Not one block cleared the absolute noise floor: the mic is muted, or it isn't the one the
    // teacher is talking into. Sending this costs a transcription to be told it was silence.
    if (prosody.peak_level < FLOOR_MIN) {
      setError("NO SOUND WAS PICKED UP — CHECK YOUR MICROPHONE.");
      return;
    }
    const merged = mergeBuffers(blocks);
    if (merged.length === 0) return;
    onUtteranceRef.current(encodeWav(downsample(merged, sampleRateRef.current)), prosody);
  }, [teardown]);

  stopRef.current = stop;

  // Keyed on state, not on the context ref: during "starting" there is no context yet, and a
  // second press there means "never mind", not "open a second microphone".
  const toggle = useCallback(() => {
    if (state === "idle") void start();
    else stop();
  }, [state, start, stop]);

  /** Drop the mic without sending anything — for leaving the room, or resetting the class. */
  const cancel = useCallback(() => {
    blocksRef.current = [];
    levelsRef.current = [];
    teardown();
    setState("idle");
    setElapsedSeconds(0);
  }, [teardown]);

  useEffect(() => {
    if (state !== "recording") return;
    const timer = window.setInterval(() => setElapsedSeconds((current) => current + 1), 1_000);
    return () => window.clearInterval(timer);
  }, [state]);

  // teardown's deps are empty on purpose: if its identity churned, this cleanup would fire and cut
  // the mic mid-recording on an unrelated re-render.
  useEffect(() => teardown, [teardown]);

  return {
    state,
    error,
    meterRef,
    elapsedSeconds,
    isRecording: state === "recording",
    /** True from the press until the recording is closed, permission prompt included. */
    isArmed: state !== "idle",
    start,
    stop,
    toggle,
    cancel,
  };
}
