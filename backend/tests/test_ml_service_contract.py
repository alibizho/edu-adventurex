"""Contract tests for the standalone GPU service and the main backend boundary."""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

import schemas as gpu_schemas
import server as gpu_server
from app import schemas as app_schemas
from app.store.db import _analysis_values, _row_to_analysis


CONTRACT_MODELS = (
    "Anomaly",
    "WordScore",
    "StudentQuestion",
    "CurriculumUpdate",
    "ChunkAnalysis",
)


def _field_contract(model: type) -> dict[str, tuple[bool, object, object]]:
    return {
        name: (field.is_required(), field.default, field.default_factory)
        for name, field in model.model_fields.items()
    }


def test_gpu_and_main_backend_schemas_share_the_same_contract():
    for model_name in CONTRACT_MODELS:
        gpu_model = getattr(gpu_schemas, model_name)
        app_model = getattr(app_schemas, model_name)
        assert _field_contract(gpu_model) == _field_contract(app_model), model_name


def test_gpu_analyze_endpoint_parses_context_and_preserves_new_fields():
    captured: dict[str, object] = {}

    class FakeEngine:
        device = "cpu-test"

        def analyze(
            self,
            audio_path,
            history,
            chunk_id,
            enable_space_c,
            overall_topic,
            curriculum_context,
            key_concepts,
        ):
            captured.update(
                audio_path=audio_path,
                history=history,
                chunk_id=chunk_id,
                enable_space_c=enable_space_c,
                overall_topic=overall_topic,
                curriculum_context=curriculum_context,
                key_concepts=key_concepts,
                existed_during_analysis=Path(audio_path).exists(),
            )
            return gpu_schemas.ChunkAnalysis(
                chunk_id=chunk_id,
                text="A particle can be in superposition.",
                confidence=0.71,
                student_question=gpu_schemas.StudentQuestion(
                    question_text="What makes the state collapse?",
                    target_concept="measurement",
                    anomaly_type="logic_error",
                ),
                curriculum_update=gpu_schemas.CurriculumUpdate(
                    added_concepts=["decoherence"]
                ),
            )

    gpu_server._engine = FakeEngine()
    try:
        response = TestClient(gpu_server.app).post(
            "/analyze",
            data={
                "chunk_id": "7",
                "history": '["first turn", "second turn"]',
                "enable_space_c": "true",
                "overall_topic": "Quantum Physics",
                "curriculum_context": "Observer effect notes",
                "key_concepts": '["measurement", "superposition"]',
            },
            files={"audio": ("turn.wav", b"RIFF-test-audio", "audio/wav")},
        )
    finally:
        gpu_server._engine = None

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["student_question"]["question_text"] == "What makes the state collapse?"
    assert body["curriculum_update"]["added_concepts"] == ["decoherence"]
    assert captured["history"] == ["first turn", "second turn"]
    assert captured["key_concepts"] == ["measurement", "superposition"]
    assert captured["overall_topic"] == "Quantum Physics"
    assert captured["curriculum_context"] == "Observer effect notes"
    assert captured["existed_during_analysis"] is True
    assert not os.path.exists(str(captured["audio_path"]))


def test_analysis_database_boundary_round_trips_new_fields():
    analysis = app_schemas.ChunkAnalysis(
        chunk_id=3,
        text="The example also introduces decoherence.",
        confidence=0.82,
        student_question=app_schemas.StudentQuestion(
            question_text="How does decoherence differ from measurement?",
            target_concept="decoherence",
            anomaly_type="beyond",
        ),
        curriculum_update=app_schemas.CurriculumUpdate(
            added_concepts=["environmental decoherence"]
        ),
    )
    values = _analysis_values("session-1", analysis)
    restored = _row_to_analysis(SimpleNamespace(**values))

    assert restored == analysis
