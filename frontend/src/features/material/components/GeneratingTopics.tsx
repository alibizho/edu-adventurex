import { useEffect, useRef, useState } from "react";

/** Milliseconds per character. Fast enough to keep up with a stage that lands every few seconds. */
const TYPE_MS = 8;

type Props = {
  /** Pipeline lines, oldest first, in the order the backend reported them. */
  lines: readonly string[];
};

/**
 * The build readout: a title and a console that types out each stage as it lands.
 *
 * The build is N+1 model calls and runs the better part of a minute. A spinner over that is
 * indistinguishable from a hang, so this shows what actually finished — the confirmed topic, the
 * class count, then each class title as the structuring call returns it and again as its material
 * is written. Every line is a real event; nothing here is on a timer pretending to be progress.
 */
export function GeneratingTopics({ lines }: Props) {
  // How much of the newest line has been revealed. Everything before it is already complete, so
  // only the tail is ever animated — a line that lands while another is typing doesn't restart it.
  const [typed, setTyped] = useState(0);
  const [shown, setShown] = useState(0);
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (shown >= lines.length) return;
    const target = lines[shown] ?? "";
    if (typed >= target.length) {
      // Line finished: commit it and start the next one on the following tick.
      const timer = window.setTimeout(() => { setShown((n) => n + 1); setTyped(0); }, 40);
      return () => window.clearTimeout(timer);
    }
    const timer = window.setTimeout(() => setTyped((n) => n + 1), TYPE_MS);
    return () => window.clearTimeout(timer);
  }, [lines, shown, typed]);

  // Follow the tail as it grows, the way a terminal does.
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
            // Lines repeat by design (a class title appears when structured and again when
            // written), so the index is the only stable identity here.
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
