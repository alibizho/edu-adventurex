export type MaterialFileKind = "document" | "image";

export type MaterialSourceKind = "text" | MaterialFileKind;

export type SelectedMaterialFile = {
  id: string;
  file: File;
  kind: MaterialFileKind;
};

export type MaterialInputStatus = "idle" | "ready" | "error" | "processing";

export type MaterialSubmissionSummary = {
  sourceKinds: MaterialSourceKind[];
  textLength: number;
  files: Array<{
    name: string;
    type: string;
    size: number;
    kind: MaterialFileKind;
  }>;
  submittedAt: string;
};

