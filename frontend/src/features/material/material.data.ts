export type MaterialPageContent = {
  title: string;
  shortcuts: readonly {
    id: "surprise" | "history" | "trending" | "physics" | "art";
    label: string;
    prompt?: string;
  }[];
  entry: {
    placeholderLines: readonly string[];
    uploadLabel: string;
    imageLabel: string;
    startLabel: string;
  };
};

export const MATERIAL_PAGE_CONTENT: MaterialPageContent = {
  title: "WHAT DO YOU WANT TO LEARN?",
  shortcuts: [
    { id: "surprise", label: "SURPRISE ME" },
    { id: "history", label: "HISTORY", prompt: "The history of quantum theory" },
    { id: "trending", label: "TRENDING", prompt: "How quantum computers use qubits" },
    { id: "physics", label: "PHYSICS", prompt: "The observer effect in quantum physics" },
    { id: "art", label: "ART", prompt: "The principles of 8-bit pixel art" },
  ],
  entry: {
    placeholderLines: [
      "TYPE KEYWORDS OR UPLOAD MATERIALS HERE (PDF, IMAGES,",
      " TEXT)...",
    ],
    uploadLabel: "UPLOAD FILE",
    imageLabel: "ADD IMAGES",
    startLabel: "START LEARNING",
  },
};

export const SURPRISE_TOPICS = [
  "How does memory shape learning?",
  "Why do octopuses solve complex problems?",
  "How does music change the way we perceive time?",
] as const;
