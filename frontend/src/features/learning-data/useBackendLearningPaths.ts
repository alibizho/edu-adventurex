import { useCallback, useEffect, useState } from "react";
import { apiMessage } from "./apiClient";
import { backendLearningDataSource } from "./backendLearningDataSource";
import type { GrowthPath, PathMemory } from "./backend.types";

export type LearningPathEntry = { path: GrowthPath; memory: PathMemory };

export function useBackendLearningPaths() {
  const [entries, setEntries] = useState<LearningPathEntry[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const reload = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const paths = await backendLearningDataSource.listPaths();
      const memories = await Promise.all(paths.map((path) => backendLearningDataSource.getMemory(path.path_id)));
      setEntries(paths.map((path, index) => ({ path, memory: memories[index] })));
    } catch (caught) {
      setError(apiMessage(caught));
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => { void reload(); }, [reload]);
  return { entries, isLoading, error, reload };
}
