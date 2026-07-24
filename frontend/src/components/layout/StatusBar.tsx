type StatusBarProps = {
  label: string;
  full?: boolean;
  meta?: string;
};

export function StatusBar({ label, full = false, meta }: StatusBarProps) {
  return (
    <footer className={`status-bar ${full ? "status-bar--full" : ""}`}>
      <div className="status-label">SYSTEM STATUS: {label}</div>
      {meta && <div className="status-meta">{meta}</div>}
      <div className="palette-indicator" aria-label="Monochrome palette">
        <i className="tone-black" />
        <i className="tone-gray" />
        <i className="tone-white" />
      </div>
    </footer>
  );
}
