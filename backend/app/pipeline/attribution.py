"""Attribution (report §4.6). Primary path: personas cite the segment ids they relied on,
so a failed question points back at the segment that broke. Stretch path (leave-one-out
ablation) is a TODO."""
import re

_CITES_RE = re.compile(r"CITES:\s*\[([0-9,\s]*)\]")


def parse_citations(answer: str) -> list[int]:
    """Pull `CITES: [1, 4]` out of a persona answer. Empty list == a gap (nothing to cite)."""
    m = _CITES_RE.search(answer)
    if not m:
        return []
    body = m.group(1).strip()
    if not body:
        return []
    return [int(x) for x in re.findall(r"\d+", body)]
