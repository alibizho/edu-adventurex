"""Small shared output-parsing helpers for LLM responses."""
import json
import re


def extract_json_array(text: str) -> list:
    """Pull the first JSON array out of model output (tolerates stray prose / code fences)."""
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []
