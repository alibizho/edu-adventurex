import { useRef, type ChangeEvent, type FormEvent } from "react";
import { ImagePlus, Search, Upload, X } from "lucide-react";
import type { MaterialPageContent } from "../material.data";
import type { MaterialFileKind, SelectedMaterialFile } from "../material.types";

type MaterialEntryPanelProps = {
  content: MaterialPageContent["entry"];
  text: string;
  files: readonly SelectedMaterialFile[];
  error: string | null;
  isProcessing: boolean;
  canSubmit: boolean;
  onTextChange: (value: string) => void;
  onFilesSelected: (kind: MaterialFileKind, files: readonly File[]) => void;
  onFileRemove: (id: string) => void;
  onSubmit: () => void;
};

function formatFileSize(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function getFileType(file: File) {
  return file.name.split(".").pop()?.toUpperCase() || file.type.toUpperCase() || "FILE";
}

export function MaterialEntryPanel({
  content,
  text,
  files,
  error,
  isProcessing,
  canSubmit,
  onTextChange,
  onFilesSelected,
  onFileRemove,
  onSubmit,
}: MaterialEntryPanelProps) {
  const documentInput = useRef<HTMLInputElement>(null);
  const imageInput = useRef<HTMLInputElement>(null);

  function handleFileChange(kind: MaterialFileKind, event: ChangeEvent<HTMLInputElement>) {
    const selectedFiles = Array.from(event.currentTarget.files ?? []);
    onFilesSelected(kind, selectedFiles);
    event.currentTarget.value = "";
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    onSubmit();
  }

  return (
    <form className="material-entry retro-panel" onSubmit={handleSubmit} aria-busy={isProcessing}>
      <div className="entry-body">
        <Search className="entry-search-icon" size={42} strokeWidth={3} aria-hidden="true" />
        <div className="entry-content">
          <label className="material-file-input" htmlFor="material-text">Learning material text</label>
          <textarea
            id="material-text"
            className="entry-textarea"
            value={text}
            onChange={(event) => onTextChange(event.currentTarget.value)}
            placeholder={content.placeholderLines.join("\n")}
            readOnly={isProcessing}
            aria-describedby={error ? "material-entry-error" : undefined}
          />

          {files.length > 0 && (
            <ul className="selected-materials" aria-label="Selected learning materials">
              {files.map(({ id, file, kind }) => (
                <li key={id} className="selected-material">
                  <span className="selected-material-kind">{kind === "image" ? "IMAGE" : "FILE"}</span>
                  <span className="selected-material-details">
                    <span className="selected-material-name" title={file.name}>{file.name}</span>
                    <span className="selected-material-meta">{getFileType(file)} / {formatFileSize(file.size)}</span>
                  </span>
                  <button
                    type="button"
                    className="remove-material"
                    onClick={() => onFileRemove(id)}
                    disabled={isProcessing}
                    aria-label={`Remove ${file.name}`}
                  >
                    <X size={20} strokeWidth={3} aria-hidden="true" />
                  </button>
                </li>
              ))}
            </ul>
          )}

          {error && <p id="material-entry-error" className="entry-error" role="alert">{error}</p>}
        </div>
      </div>

      <input
        ref={documentInput}
        className="material-file-input"
        type="file"
        accept=".pdf,.txt,.md,application/pdf,text/plain,text/markdown"
        multiple
        disabled={isProcessing}
        aria-label="Choose PDF or text files"
        onChange={(event) => handleFileChange("document", event)}
      />
      <input
        ref={imageInput}
        className="material-file-input"
        type="file"
        accept=".png,.jpg,.jpeg,.webp,image/png,image/jpeg,image/webp"
        multiple
        disabled={isProcessing}
        aria-label="Choose image files"
        onChange={(event) => handleFileChange("image", event)}
      />

      <div className="entry-actions">
        <div className="entry-secondary-actions">
          <button type="button" className="outline-action" disabled={isProcessing} onClick={() => documentInput.current?.click()}>
            <Upload size={24} strokeWidth={2.5} aria-hidden="true" />{content.uploadLabel}
          </button>
          <button type="button" className="outline-action" disabled={isProcessing} onClick={() => imageInput.current?.click()}>
            <ImagePlus size={24} strokeWidth={2.5} aria-hidden="true" />{content.imageLabel}
          </button>
        </div>
        <button type="submit" className="solid-action" disabled={!canSubmit}>
          {isProcessing ? "PREPARING..." : content.startLabel}
        </button>
      </div>
    </form>
  );
}
