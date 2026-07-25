"""Shared multipart form-field parsing.

Both audio routes take JSON-encoded arrays as form fields (`history`, `key_concepts`). They had
separate copies of the same try/except, which drifted: one degraded to `[]`, the other let a
malformed value 422 the whole request. One definition, one behaviour.
"""
import json


def json_string_list(raw: str) -> list[str]:
    """Parse a JSON array form field into a list of strings.

    Degrades to `[]` rather than rejecting the request: these fields carry optional context (prior
    transcripts, known concepts), so a malformed one should cost the caller that context, not the
    utterance they just recorded.
    """
    try:
        parsed = json.loads(raw) if raw else []
    except json.JSONDecodeError:
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []
