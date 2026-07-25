import type { ConceptNodeConfig } from "./concepts.types";

export const CONCEPTS: readonly ConceptNodeConfig[] = [
  { id: "fundamentals", label: "FUNDAMENTALS", shape: "square", x: 22, y: 27, icon: "solid" },
  { id: "ethics", label: "ETHICS", shape: "square small", x: 48, y: 16 },
  { id: "advanced-logic", label: "ADVANCED\nLOGIC", shape: "square", x: 69, y: 21, icon: "outline" },
  { id: "practical-application", label: "PRACTICAL\nAPPLICATION", shape: "wide", x: 75, y: 53 },
  { id: "case-studies", label: "CASE STUDIES", shape: "square", x: 17, y: 72, icon: "grid" },
  { id: "subject", label: "QUANTUM PHYSICS", shape: "circle hero", x: 45, y: 49 },
  { id: "tools", label: "TOOLS", shape: "circle small-circle", x: 38, y: 64 },
  { id: "history", label: "HISTORY", shape: "circle medium-circle", x: 63, y: 77 },
];
