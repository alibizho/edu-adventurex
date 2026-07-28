import { useEffect, useId, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import "katex/dist/katex.min.css";

const MERMAID_CONFIG = {
  startOnLoad: false,
  securityLevel: "strict",
  theme: "neutral",
  fontFamily: '"Space Mono", "Courier New", monospace',
  flowchart: { htmlLabels: true, padding: 12, wrappingWidth: 240 },
  sequence: { wrap: true },
} as const;

const CODE_SPAN = /(```[\s\S]*?```|`[^`\n]*`)/g;

function normalizeMath(source: string) {
  return source
    .split(CODE_SPAN)
    .map((part, index) => (index % 2 === 1 ? part : part
      .replace(/\\\[([\s\S]+?)\\\]/g, (_, body: string) => `\n\n$$\n${body.trim()}\n$$\n\n`)
      .replace(/\\\(([\s\S]+?)\\\)/g, (_, body: string) => `$${body.trim()}$`)))
    .join("");
}

function releaseSvgSize(svg: string) {
  if (!svg.includes("viewBox")) return svg;
  return svg.replace(/<svg[^>]*>/, (tag) => tag.replace(/\s(?:style|width|height)="[^"]*"/g, ""));
}

function MermaidBlock({ source }: { source: string }) {
  const id = `mermaid-${useId().replaceAll(":", "")}`;
  const host = useRef<HTMLDivElement>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let active = true;

    async function draw() {
      const { default: mermaid } = await import("mermaid");
      await document.fonts?.ready;
      if (!active) return;

      mermaid.initialize(MERMAID_CONFIG);
      const { svg } = await mermaid.render(id, source);
      if (active && host.current) host.current.innerHTML = releaseSvgSize(svg);
    }

    draw().catch(() => { if (active) setError(true); });
    return () => { active = false; };
  }, [id, source]);

  return error
    ? <pre className="notes-code"><code>{source}</code></pre>
    : <div ref={host} className="notes-mermaid" role="img" aria-label="Learning diagram" />;
}

export function MarkdownNotes({ source }: { source: string }) {
  const markdown = useMemo(() => normalizeMath(source), [source]);

  return (
    <div className="markdown-notes">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[[rehypeKatex, { throwOnError: false, strict: false }]]}
        components={{
          code({ className, children, ...props }) {
            const language = /language-(\w+)/.exec(className ?? "")?.[1];
            const text = String(children).replace(/\n$/, "");
            if (language === "mermaid") return <MermaidBlock source={text} />;

            return <code className={className} {...props}>{children}</code>;
          },
        }}
      >
        {markdown}
      </ReactMarkdown>
    </div>
  );
}
