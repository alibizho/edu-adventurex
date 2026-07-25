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
    </footer>
  );
}
