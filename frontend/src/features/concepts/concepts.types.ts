import type { MaterialSubmissionSummary } from "../material/material.types";

export type ConceptId =
  | "fundamentals"
  | "ethics"
  | "advanced-logic"
  | "practical-application"
  | "case-studies"
  | "subject"
  | "tools"
  | "history";

export type ConceptIcon = "solid" | "outline" | "grid";

export type ConceptNodeShape =
  | "square"
  | "square small"
  | "wide"
  | "circle hero"
  | "circle small-circle"
  | "circle medium-circle";

export type ConceptNodeConfig = {
  id: ConceptId;
  label: string;
  shape: ConceptNodeShape;
  x: number;
  y: number;
  icon?: ConceptIcon;
};

export type ConceptSelectionSummary = {
  id: ConceptId;
  label: string;
  selectedAt: string;
};

export type ConceptPageRouteState = {
  material?: MaterialSubmissionSummary;
};

export type StudyRouteState = ConceptPageRouteState & {
  concept: ConceptSelectionSummary;
};
