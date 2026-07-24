import { apiRequest } from "./apiClient";
import type {
  AnalysisStatus,
  AudioClassTeachResponse,
  BackendHealth,
  ClassTeachResponse,
  ClassUnit,
  GrowthPath,
  MaterialExtraction,
  PathMemory,
  SessionSnapshot,
  TopicScope,
} from "./backend.types";

function segment(value: string) {
  return encodeURIComponent(value);
}

export const backendLearningDataSource = {
  health: () => apiRequest<BackendHealth>("/health", {}, 6_000),
  confusionHealth: () => apiRequest<{ reachable: boolean; error?: string }>("/confusion/health", {}, 7_000),

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

  teachAudio(pathId: string, classId: string, audio: Blob, chunkId: number, history: string[]) {
    const form = new FormData();
    form.append("chunk_id", String(chunkId));
    form.append("history", JSON.stringify(history));
    form.append("audio", audio, `chunk-${chunkId}.wav`);
    return apiRequest<AudioClassTeachResponse>(
      `/plan/${segment(pathId)}/class/${segment(classId)}/teach/audio-turn`,
      { method: "POST", body: form },
      120_000,
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
