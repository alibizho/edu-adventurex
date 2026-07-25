from io import BytesIO

import pytest
from PIL import Image

from app import materials

def test_txt_and_markdown_are_combined_without_persisting_bytes():
    result = materials.extract_materials([
        ("lesson.txt", "text/plain", b"Energy is conserved."),
        ("notes.md", "text/markdown", b"# Momentum\nMass times velocity."),
    ])
    assert "Energy is conserved" in result.material_text
    assert "Mass times velocity" in result.material_text
    assert [item.name for item in result.files] == ["lesson.txt", "notes.md"]
    assert result.truncated is False

def test_image_uses_english_ocr(monkeypatch):
    stream = BytesIO()
    Image.new("RGB", (8, 8), "white").save(stream, format="PNG")
    monkeypatch.setattr(materials, "_ocr_image", lambda image: "Printed English text")
    result = materials.extract_materials([
        ("scan.png", "image/png", stream.getvalue()),
    ])
    assert "Printed English text" in result.material_text
    assert result.files[0].extracted_characters == len("Printed English text")

def test_limits_unsupported_and_duplicate_files_are_atomic():
    with pytest.raises(materials.MaterialExtractionError, match="UNSUPPORTED"):
        materials.extract_materials([("payload.exe", "application/octet-stream", b"x")])
    with pytest.raises(materials.MaterialExtractionError, match="DUPLICATE"):
        materials.extract_materials([
            ("same.txt", "text/plain", b"one"),
            ("same.txt", "text/plain", b"two"),
        ])
    with pytest.raises(materials.MaterialExtractionError, match="MAXIMUM"):
        materials.extract_materials([
            (f"{index}.txt", "text/plain", b"x") for index in range(11)
        ])

def test_material_text_is_truncated_to_generation_limit():
    payload = b"a" * (materials.MAX_EXTRACTED_CHARACTERS + 500)
    result = materials.extract_materials([("long.txt", "text/plain", payload)])
    assert result.truncated is True
    assert len(result.material_text) == materials.MAX_EXTRACTED_CHARACTERS
    assert any("TRUNCATED" in warning for warning in result.warnings)

def test_dangerous_filename_is_reduced_to_basename():
    result = materials.extract_materials([
        ("../../lesson.md", "text/markdown", b"safe"),
    ])
    assert result.files[0].name == "lesson.md"
