import { useEffect, useRef, useState } from "react";

const TYPE_MS = 8;

type Props = {
  lines: readonly string[];
};

export function GeneratingTopics({ lines }: Props) {
  const [typed, setTyped] = useState(0);
  const [shown, setShown] = useState(0);
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (shown >= lines.length) return;
    const target = lines[shown] ?? "";
    if (typed >= target.length) {
      const timer = window.setTimeout(() => { setShown((n) => n + 1); setTyped(0); }, 40);
      return () => window.clearTimeout(timer);
    }
    const timer = window.setTimeout(() => setTyped((n) => n + 1), TYPE_MS);
    return () => window.clearTimeout(timer);
  }, [lines, shown, typed]);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [typed, shown]);

  const complete = lines.slice(0, shown);
  const partial = shown < lines.length ? (lines[shown] ?? "").slice(0, typed) : null;

  return (
    <section className="generating" aria-live="polite" aria-busy="true">
      <h1 className="boxed-title generating-title">GENERATING TOPICS...</h1>

      <div className="retro-panel generating-console">
        <header className="generating-console-bar">
          <span>PROCESS PIPELINE: V1.0.4</span>

          <span className="generating-console-dots" aria-hidden="true" />
        </header>

        <div className="generating-log" ref={logRef}>
          {complete.map((line, index) => (
            <p key={`${index}-${line}`} className={line.startsWith("  ") ? "is-indented" : ""}>
              {line || " "}
            </p>

          ))}
          {partial !== null && (
            <p className={partial.startsWith("  ") ? "is-indented" : ""}>{partial}</p>

          )}
          <span className="generating-caret" aria-hidden="true" />
        </div>

      </div>

    </section>

  );
}
