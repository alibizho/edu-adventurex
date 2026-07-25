import { useRef, useState } from "react";
import { apiMessage } from "../learning-data/apiClient";
import { backendLearningDataSource } from "../learning-data/backendLearningDataSource";
import type { BuildEvent, GrowthPath, ScopeSuggestion } from "../learning-data/backend.types";
import type {
  MaterialFileKind,
  MaterialInputStatus,
  SelectedMaterialFile,
} from "./material.types";

const MAX_FILE_COUNT = 10;
const MAX_FILE_SIZE = 10 * 1024 * 1024;
const MAX_TOTAL_SIZE = 30 * 1024 * 1024;

const ACCEPTED_EXTENSIONS: Record<MaterialFileKind, ReadonlySet<string>> = {
  document: new Set(["pdf", "txt", "md"]),
  image: new Set(["png", "jpg", "jpeg", "webp"]),
};

function getFileExtension(fileName: string) {
  return fileName.split(".").pop()?.toLowerCase() ?? "";
}

function getFileId(file: File) {
  return [file.name, file.size, file.type, file.lastModified].join("-");
}

function validateFiles(
  kind: MaterialFileKind,
  incomingFiles: readonly File[],
  selectedFiles: readonly SelectedMaterialFile[],
) {
  if (selectedFiles.length + incomingFiles.length > MAX_FILE_COUNT) {
    return `A MAXIMUM OF ${MAX_FILE_COUNT} FILES IS ALLOWED.`;
  }

  const knownIds = new Set(selectedFiles.map(({ id }) => id));

  for (const file of incomingFiles) {
    if (!ACCEPTED_EXTENSIONS[kind].has(getFileExtension(file.name))) {
      return `UNSUPPORTED FILE TYPE: ${file.name}`;
    }

    if (file.size > MAX_FILE_SIZE) {
      return `FILE EXCEEDS 10 MB: ${file.name}`;
    }

    const id = getFileId(file);
    if (knownIds.has(id)) {
      return `FILE ALREADY ADDED: ${file.name}`;
    }
    knownIds.add(id);
  }

  const currentTotal = selectedFiles.reduce((total, { file }) => total + file.size, 0);
  const incomingTotal = incomingFiles.reduce((total, file) => total + file.size, 0);
  if (currentTotal + incomingTotal > MAX_TOTAL_SIZE) {
    return "TOTAL FILE SIZE CANNOT EXCEED 30 MB.";
  }

  return null;
}

type UseMaterialInputOptions = {
  onPrepared: (path: GrowthPath) => void;
};

function pipelineLines(event: BuildEvent): string[] {
  switch (event.stage) {
    case "topic":
      return [
        `[PIPELINE] CONFIRMED TOPIC: ${event.topic.toUpperCase()}`,
        `[PIPELINE] CLASSES: ${event.classes}`,
        "",
      ];
    case "structuring":
      return [`[AGENT 2] STRUCTURING ${event.classes} CLASSES FOR: ${event.topic.toUpperCase()}`, ""];
    case "class":
      return [`  [${event.index}/${event.total}] ${event.title.toUpperCase()}`];
    case "writing":
      return ["", `[AGENT 3] WRITING NOTES FOR ${event.total} CLASSES...`];
    case "written":
      return [
        `  [${event.index}/${event.total}] ${event.title.toUpperCase()}${event.ok ? "" : " -- FAILED, WILL RETRY ON OPEN"}`,
      ];
    case "done":
      return ["", `[PIPELINE] COURSE READY: ${event.path.path_id}`];
    case "error":
      return ["", `[PIPELINE] ERROR: ${event.message}`];
  }
}

export function useMaterialInput({ onPrepared }: UseMaterialInputOptions) {
  const [text, setText] = useState("");
  const [files, setFiles] = useState<SelectedMaterialFile[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [scopeSuggestions, setScopeSuggestions] = useState<ScopeSuggestion[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [pipeline, setPipeline] = useState<string[]>([]);
  const preparedMaterial = useRef<{ originalInput: string; materialText: string | null } | null>(null);

  const hasMaterial = text.trim().length > 0 || files.length > 0;
  const status: MaterialInputStatus = isProcessing
    ? "processing"
    : error
      ? "error"
      : hasMaterial
        ? "ready"
        : "idle";

  function updateText(value: string) {
    if (isProcessing) return;
    setText(value);
    setError(null);
    setScopeSuggestions([]);
    preparedMaterial.current = null;
  }

  function addFiles(kind: MaterialFileKind, incomingFiles: readonly File[]) {
    if (isProcessing || incomingFiles.length === 0) return;

    const validationError = validateFiles(kind, incomingFiles, files);
    if (validationError) {
      setError(validationError);
      return;
    }

    setFiles((currentFiles) => [
      ...currentFiles,
      ...incomingFiles.map((file) => ({ id: getFileId(file), file, kind })),
    ]);
    setError(null);
    setScopeSuggestions([]);
    preparedMaterial.current = null;
  }

  function removeFile(id: string) {
    if (isProcessing) return;
    setFiles((currentFiles) => currentFiles.filter((item) => item.id !== id));
    setError(null);
    setScopeSuggestions([]);
    preparedMaterial.current = null;
  }

  async function buildConfirmedTopic(topic: string, suggestedClasses: number) {
    const prepared = preparedMaterial.current;
    if (!prepared || isProcessing) return;
    setIsProcessing(true);
    setError(null);
    setPipeline([]);
    try {
      const path = await backendLearningDataSource.buildPlanStream({
        originalInput: prepared.originalInput,
        confirmedTopic: topic,
        numClasses: suggestedClasses,
        materialText: prepared.materialText,
      }, (event) => setPipeline((current) => [...current, ...pipelineLines(event)]));
      onPrepared(path);
    } catch (caught) {
      setError(apiMessage(caught));
      setPipeline([]);
      setIsProcessing(false);
    }
  }

  async function submit() {
    if (!hasMaterial || isProcessing) return;
    setError(null);
    setWarnings([]);
    setScopeSuggestions([]);
    setIsProcessing(true);
    try {
      const extracted = files.length
        ? await backendLearningDataSource.extractMaterials(files.map(({ file }) => file))
        : null;
      const typedText = text.trim();
      const materialText = [typedText, extracted?.material_text]
        .filter(Boolean)
        .join("\n\n") || null;
      const originalInput = typedText || "Create a learning path from the uploaded material";
      preparedMaterial.current = { originalInput, materialText };
      setWarnings(extracted?.warnings ?? []);
      const scope = await backendLearningDataSource.scopeTopic(originalInput, materialText);
      if (scope.is_broad && scope.suggestions.length) {
        setScopeSuggestions(scope.suggestions);
        setIsProcessing(false);
        return;
      }
      await buildConfirmedTopic(scope.confirmed_topic, scope.suggested_classes);
    } catch (caught) {
      setError(apiMessage(caught));
      setIsProcessing(false);
    }
  }

  return {
    text,
    files,
    error,
    warnings,
    scopeSuggestions,
    pipeline,
    isProcessing,
    canSubmit: hasMaterial && !isProcessing,
    status,
    updateText,
    addFiles,
    removeFile,
    submit,
    buildConfirmedTopic,
  };
}
