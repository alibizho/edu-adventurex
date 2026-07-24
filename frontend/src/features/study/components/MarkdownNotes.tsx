import { useEffect, useId, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

function MermaidBlock({ source }: { source: string }) {
  const id = `mermaid-${useId().replaceAll(":", "")}`;
  const host = useRef<HTMLDivElement>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let active = true;
    import("mermaid")
      .then(async ({ default: mermaid }) => {
        mermaid.initialize({ startOnLoad: false, securityLevel: "strict", theme: "neutral" });
        const { svg } = await mermaid.render(id, source);
        if (active && host.current) host.current.innerHTML = svg;
      })
      .catch(() => { if (active) setError(true); });
    return () => { active = false; };
  }, [id, source]);

  return error
    ? <pre className="notes-code"><code>{source}</code></pre>
    : <div ref={host} className="notes-mermaid" role="img" aria-label="Learning diagram" />;
}

export function MarkdownNotes({ source }: { source: string }) {
  return (
    <div className="markdown-notes">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          code({ className, children, ...props }) {
            const language = /language-(\w+)/.exec(className ?? "")?.[1];
            const text = String(children).replace(/\n$/, "");
            if (language === "mermaid") return <MermaidBlock source={text} />;
            return <code className={className} {...props}>{children}</code>;
          },
        }}
      >
        {source}
      </ReactMarkdown>
    </div>
  );
}
