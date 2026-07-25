import { useCallback, useEffect, useRef, useState } from "react";
import { downsample, encodeWav, mergeBuffers } from "./useWavRecorder";

const MAX_CAPTURE_MS = 5 * 60_000;
const MIN_CAPTURE_MS = 800;
const FLOOR_MIN = 0.006;
const BLOCK_SIZE = 4096;

const HESITATION_PAUSE_MS = 350;

const METER_GAIN = 900;

export type RecorderState = "idle" | "starting" | "recording";

export type MeterRef = (node: HTMLDivElement | null) => void;

export type SpeechProsody = {
  speech_ms: number;
  total_ms: number;
  pause_count: number;
  longest_pause_ms: number;
  mean_level: number;
  peak_level: number;
};

export function measureProsody(levels: readonly number[], blockMs: number): SpeechProsody {
  const round = (value: number) => Math.round(value);
  if (levels.length === 0) {
    return { speech_ms: 0, total_ms: 0, pause_count: 0, longest_pause_ms: 0, mean_level: 0, peak_level: 0 };
  }

  const sorted = [...levels].sort((a, b) => a - b);
  const loudest = sorted[sorted.length - 1];
  const room = sorted[Math.floor(sorted.length * 0.05)];
  const threshold = Math.min(Math.max(room * 3, FLOOR_MIN), loudest * 0.35);

  const first = levels.findIndex((level) => level > threshold);
  const last = levels.length - 1 - [...levels].reverse().findIndex((level) => level > threshold);
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

export function formatElapsed(seconds: number) {
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
}

type Options = {
  onUtterance: (audio: Blob, prosody: SpeechProsody) => void;
};

export function usePressToTalkRecorder({ onUtterance }: Options) {
  const [state, setState] = useState<RecorderState>("idle");
  const [error, setError] = useState<string | null>(null);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  const contextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);

  const levelRef = useRef(0);
  const meterNodeRef = useRef<HTMLDivElement | null>(null);
  const meterFrameRef = useRef(0);
  const paintedPctRef = useRef(-1);

  const startTokenRef = useRef(0);

  const blocksRef = useRef<Float32Array[]>([]);
  const levelsRef = useRef<number[]>([]);
  const sampleRateRef = useRef(48_000);
  const blockMsRef = useRef((BLOCK_SIZE / 48_000) * 1000);
  const onUtteranceRef = useRef(onUtterance);
  onUtteranceRef.current = onUtterance;

  const stopRef = useRef<() => void>(() => {});

  const paintMeter = useCallback(() => {
    const node = meterNodeRef.current;
    if (!node) return;
    const pct = Math.min(100, Math.round(levelRef.current * METER_GAIN));
    if (pct === paintedPctRef.current) return;
    paintedPctRef.current = pct;
    node.style.width = `${pct}%`;
  }, []);

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
    startTokenRef.current += 1;
    if (meterFrameRef.current) cancelAnimationFrame(meterFrameRef.current);
    meterFrameRef.current = 0;
    levelRef.current = 0;
    paintedPctRef.current = -1;
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
    const token = (startTokenRef.current += 1);
    setError(null);
    setElapsedSeconds(0);
    setState("starting");
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
    } catch {
      if (startTokenRef.current === token) {
        setError("MICROPHONE ACCESS WAS DENIED OR IS UNAVAILABLE.");
        setState("idle");
      }
      return;
    }

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
    if (prosody.peak_level < FLOOR_MIN) {
      setError("NO SOUND WAS PICKED UP — CHECK YOUR MICROPHONE.");
      return;
    }
    const merged = mergeBuffers(blocks);
    if (merged.length === 0) return;
    onUtteranceRef.current(encodeWav(downsample(merged, sampleRateRef.current)), prosody);
  }, [teardown]);

  stopRef.current = stop;

  const toggle = useCallback(() => {
    if (state === "idle") void start();
    else stop();
  }, [state, start, stop]);

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

  useEffect(() => teardown, [teardown]);

  return {
    state,
    error,
    meterRef,
    elapsedSeconds,
    isRecording: state === "recording",
    isArmed: state !== "idle",
    start,
    stop,
    toggle,
    cancel,
  };
}
