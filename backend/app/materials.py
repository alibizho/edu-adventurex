from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pypdfium2 as pdfium
import pytesseract
from PIL import Image, UnidentifiedImageError
from pypdf import PdfReader

from .schemas import MaterialExtractionResponse, MaterialFileSummary

MAX_FILES = 10
MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_TOTAL_SIZE = 30 * 1024 * 1024
MAX_EXTRACTED_CHARACTERS = 120_000
MAX_PDF_PAGES = 40
ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf", ".png", ".jpg", ".jpeg", ".webp"}

class MaterialExtractionError(ValueError):
    pass

def _clean_text(value: str) -> str:
    return "\n".join(line.rstrip() for line in value.replace("\x00", "").splitlines()).strip()

def _ocr_image(image: Image.Image) -> str:
    return _clean_text(pytesseract.image_to_string(image.convert("RGB"), lang="eng"))

def _extract_pdf(data: bytes, name: str, warnings: list[str]) -> str:
    try:
        reader = PdfReader(BytesIO(data))
    except Exception as exc:
        raise MaterialExtractionError(f"INVALID PDF: {name}") from exc

    texts: list[str] = []
    page_count = min(len(reader.pages), MAX_PDF_PAGES)
    if len(reader.pages) > MAX_PDF_PAGES:
        warnings.append(f"{name}: ONLY THE FIRST {MAX_PDF_PAGES} PAGES WERE READ.")

    pdf_document = None
    for page_index in range(page_count):
        page_text = _clean_text(reader.pages[page_index].extract_text() or "")
        if len(page_text) < 24:
            try:
                pdf_document = pdf_document or pdfium.PdfDocument(data)
                bitmap = pdf_document[page_index].render(scale=2)
                page_text = _ocr_image(bitmap.to_pil())
                if page_text:
                    warnings.append(f"{name}: PAGE {page_index + 1} USED ENGLISH OCR.")
            except Exception:
                warnings.append(f"{name}: PAGE {page_index + 1} COULD NOT BE OCR PROCESSED.")
        if page_text:
            texts.append(page_text)

    return "\n\n".join(texts)

def _extract_one(name: str, data: bytes, warnings: list[str]) -> str:
    extension = Path(name).suffix.lower()
    if extension in {".txt", ".md"}:
        try:
            return _clean_text(data.decode("utf-8-sig"))
        except UnicodeDecodeError:
            warnings.append(f"{name}: UTF-8 FAILED; LATIN-1 FALLBACK WAS USED.")
            return _clean_text(data.decode("latin-1"))
    if extension == ".pdf":
        return _extract_pdf(data, name, warnings)
    try:
        with Image.open(BytesIO(data)) as image:
            return _ocr_image(image)
    except (UnidentifiedImageError, OSError) as exc:
        raise MaterialExtractionError(f"INVALID IMAGE: {name}") from exc

def extract_materials(files: list[tuple[str, str, bytes]]) -> MaterialExtractionResponse:
    if not files:
        return MaterialExtractionResponse(material_text="", files=[])
    if len(files) > MAX_FILES:
        raise MaterialExtractionError(f"A MAXIMUM OF {MAX_FILES} FILES IS ALLOWED.")

    total_size = sum(len(data) for _, _, data in files)
    if total_size > MAX_TOTAL_SIZE:
        raise MaterialExtractionError("TOTAL FILE SIZE CANNOT EXCEED 30 MB.")

    seen: set[tuple[str, int]] = set()
    for raw_name, _, data in files:
        name = Path(raw_name).name
        if Path(name).suffix.lower() not in ALLOWED_EXTENSIONS:
            raise MaterialExtractionError(f"UNSUPPORTED FILE TYPE: {name}")
        if len(data) > MAX_FILE_SIZE:
            raise MaterialExtractionError(f"FILE EXCEEDS 10 MB: {name}")
        identity = (name.casefold(), len(data))
        if identity in seen:
            raise MaterialExtractionError(f"DUPLICATE FILE: {name}")
        seen.add(identity)

    warnings: list[str] = []
    summaries: list[MaterialFileSummary] = []
    sections: list[str] = []
    for raw_name, media_type, data in files:
        name = Path(raw_name).name
        text = _extract_one(name, data, warnings)
        if not text:
            warnings.append(f"{name}: NO READABLE ENGLISH TEXT WAS FOUND.")
        else:
            sections.append(f"--- SOURCE: {name} ---\n{text}")
        summaries.append(MaterialFileSummary(
            name=name,
            media_type=media_type or "application/octet-stream",
            size=len(data),
            extracted_characters=len(text),
        ))

    combined = "\n\n".join(sections)
    truncated = len(combined) > MAX_EXTRACTED_CHARACTERS
    if truncated:
        combined = combined[:MAX_EXTRACTED_CHARACTERS]
        warnings.append("EXTRACTED MATERIAL WAS TRUNCATED TO 120000 CHARACTERS.")
    return MaterialExtractionResponse(
        material_text=combined,
        files=summaries,
        warnings=warnings,
        truncated=truncated,
    )
