import re

_CITES_RE = re.compile(r"CITES:\s*\[([0-9,\s]*)\]")

def parse_citations(answer: str) -> list[int]:
    m = _CITES_RE.search(answer)
    if not m:
        return []
    body = m.group(1).strip()
    if not body:
        return []
    return [int(x) for x in re.findall(r"\d+", body)]
