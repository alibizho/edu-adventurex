import { Route, Routes } from "react-router-dom";
import { ConceptPage } from "../features/concepts/ConceptPage";
import { MaterialPage } from "../features/material/MaterialPage";
import { NotFoundPage } from "../features/not-found/NotFoundPage";
import { KnowledgeMapPage } from "../features/knowledge-map/KnowledgeMapPage";
import { StudyPage } from "../features/study/StudyPage";
import { SummaryPage } from "../features/summary/SummaryPage";
import { ROUTES } from "./routes";
import { ScrollToTop } from "./ScrollToTop";

export function App() {
  return (
    <>
      <ScrollToTop />
      <Routes>
        <Route path={ROUTES.material} element={<MaterialPage />} />

        <Route path={ROUTES.concepts} element={<ConceptPage />} />

        <Route path={ROUTES.study} element={<StudyPage />} />

        <Route path={ROUTES.summary} element={<SummaryPage />} />

        <Route path={ROUTES.map} element={<KnowledgeMapPage />} />

        <Route path="*" element={<NotFoundPage />} />

      </Routes>

    </>

  );
}
