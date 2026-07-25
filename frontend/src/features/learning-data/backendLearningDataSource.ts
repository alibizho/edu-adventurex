import { ApiError, apiRequest, apiStream } from "./apiClient";
import type {
  AnalysisStatus,
  AudioClassTeachResponse,
  BackendHealth,
  BuildEvent,
  ClassTeachResponse,
  ClassUnit,
  GrowthPath,
  MaterialExtraction,
  PathMemory,
  SessionSnapshot,
  TopicScope,
} from "./backend.types";
import type { SpeechProsody } from "../study/usePressToTalkRecorder";

function segment(value: string) {
  return encodeURIComponent(value);
}

export const backendLearningDataSource = {
  health: () => apiRequest<BackendHealth>("/health", {}, 6_000),
  // Must stay above the backend's own ml_service_health_timeout: this probe crosses a tunnel to a
  // rented GPU box and measures 4-8s. At 7s it timed out intermittently and the class dropped to
  // "GPU VOICE ANALYSIS OFFLINE" with the engine up and healthy.
  confusionHealth: () => apiRequest<{ reachable: boolean; error?: string }>("/confusion/health", {}, 25_000),

  extractMaterials(files: readonly File[]) {
    const form = new FormData();
    files.forEach((file) => form.append("files", file));
    return apiRequest<MaterialExtraction>("/materials/extract", { method: "POST", body: form }, 120_000);
  },

  scopeTopic(originalInput: string, materialText: string | null) {
    return apiRequest<TopicScope>("/plan/scope", {
      method: "POST",
      body: JSON.stringify({ original_input: originalInput, material_text: materialText }),
    }, 120_000);
  },

  buildPlan(input: { originalInput: string; confirmedTopic: string; numClasses: number; materialText: string | null }) {
    return apiRequest<GrowthPath>("/plan/build", {
      method: "POST",
      body: JSON.stringify({
        original_input: input.originalInput,
        confirmed_topic: input.confirmedTopic,
        num_classes: input.numClasses,
        material_text: input.materialText,
      }),
    }, 180_000);
  },

  /**
   * The same build, reported stage by stage. Resolves with the stored path once the stream ends.
   *
   * A failure arrives as an `error` event rather than a status code — the response has already
   * begun by the time anything can go wrong — so it is turned back into a thrown ApiError here
   * and callers can treat both paths the same.
   */
  async buildPlanStream(
    input: { originalInput: string; confirmedTopic: string; numClasses: number; materialText: string | null },
    onEvent: (event: BuildEvent) => void,
  ): Promise<GrowthPath> {
    let built: GrowthPath | null = null;
    let failure: string | null = null;
    await apiStream<BuildEvent>("/plan/build/stream", {
      method: "POST",
      body: JSON.stringify({
        original_input: input.originalInput,
        confirmed_topic: input.confirmedTopic,
        num_classes: input.numClasses,
        material_text: input.materialText,
      }),
    }, (event) => {
      if (event.stage === "done") built = event.path;
      if (event.stage === "error") failure = event.message;
      onEvent(event);
    });
    if (failure) throw new ApiError(failure, 502);
    // Reached the end without either: the connection dropped mid-build.
    if (!built) throw new ApiError("THE BUILD ENDED BEFORE THE COURSE WAS READY.", 0);
    return built;
  },

  listPaths: () => apiRequest<GrowthPath[]>("/plan", {}, 30_000),
  getPath: (pathId: string) => apiRequest<GrowthPath>(`/plan/${segment(pathId)}`),
  getMemory: (pathId: string) => apiRequest<PathMemory>(`/plan/${segment(pathId)}/memory`),
  generateNotes: (pathId: string, classId: string) => apiRequest<ClassUnit>(
    `/plan/${segment(pathId)}/class/${segment(classId)}/notes`,
    { method: "POST" },
    180_000,
  ),

  teachText(pathId: string, classId: string, latestUtterance: string) {
    return apiRequest<ClassTeachResponse>(
      `/plan/${segment(pathId)}/class/${segment(classId)}/teach/turn`,
      { method: "POST", body: JSON.stringify({ latest_utterance: latestUtterance }) },
      120_000,
    );
  },

  /**
   * `silent` is the live-classroom mode: the chunk is still transcribed, stored and analyzed, but
   * the class only speaks if a question fires. Pass false in the one-to-one zoom, where the
   * student is answering you directly and is expected to say something back.
   */
  teachAudio(
    pathId: string,
    classId: string,
    audio: Blob,
    chunkId: number,
    history: string[],
    silent = false,
    prosody?: SpeechProsody,
    answeringQuestionId?: number,
  ) {
    const form = new FormData();
    form.append("chunk_id", String(chunkId));
    form.append("history", JSON.stringify(history));
    form.append("silent", String(silent));
    // Answering one student face to face: the backend grades this against the question's answer
    // key and may come back with a follow-up instead of accepting whatever was said.
    if (answeringQuestionId !== undefined) {
      form.append("answering_question_id", String(answeringQuestionId));
    }
    // How it sounded. The GPU can't see hesitation that runs through a whole utterance, so the
    // recorder measures pauses and dead air directly (see usePressToTalkRecorder.ts).
    if (prosody) {
      Object.entries(prosody).forEach(([key, value]) => form.append(key, String(value)));
    }
    form.append("audio", audio, `chunk-${chunkId}.wav`);
    return apiRequest<AudioClassTeachResponse>(
      `/plan/${segment(pathId)}/class/${segment(classId)}/teach/audio-turn`,
      { method: "POST", body: form },
      120_000,
    );
  },

  /** Records the teacher's answer so the question agent won't probe the same gap again. */
  answerQuestion(sessionId: string, questionId: number, answer: string) {
    return apiRequest<{ recorded: boolean }>("/questions/answer", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, question_id: questionId, answer }),
    }, 30_000);
  },

  /** Wipe this class's session (speech, analyses, questions, progress) and hand back the memory. */
  resetClass(pathId: string, classId: string) {
    return apiRequest<PathMemory>(
      `/plan/${segment(pathId)}/class/${segment(classId)}/reset`,
      { method: "POST" },
      30_000,
    );
  },

  endClass(pathId: string, classId: string, completionMode: "self-teaching" | "guided-explanation") {
    return apiRequest<PathMemory>(
      `/plan/${segment(pathId)}/class/${segment(classId)}/end`,
      { method: "POST", body: JSON.stringify({ completion_mode: completionMode }) },
      30_000,
    );
  },

  startAnalysis: (sessionId: string) => apiRequest<AnalysisStatus>(
    `/analysis/${segment(sessionId)}`,
    { method: "POST" },
    15_000,
  ),
  getAnalysis: (sessionId: string) => apiRequest<AnalysisStatus>(`/analysis/${segment(sessionId)}`),
  getSession: (sessionId: string) => apiRequest<SessionSnapshot>(`/sessions/${segment(sessionId)}`),
};
