from .schemas import (Arm, ChunkAnalysis, FusionResult, QuestionDelta, Score,
                      SegmentFusion)

DISTURBANCE_HIGH = 0.5

def _segment_deltas(scores: list[Score], per_question: list[QuestionDelta]) -> dict[int, float]:
    qdelta = {d.question_id: d.delta for d in per_question}
    seg_qs: dict[int, set[int]] = {}
    for s in scores:
        if s.arm == Arm.taught:
            for seg in s.cited_segment_ids:
                seg_qs.setdefault(seg, set()).add(s.question_id)
    out: dict[int, float] = {}
    for seg, qids in seg_qs.items():
        ds = [qdelta[q] for q in qids if q in qdelta]
        if ds:
            out[seg] = sum(ds) / len(ds)
    return out

def _quadrant(disturbance: float, delta: float | None) -> str:
    if delta is None:
        return "unknown"
    high = disturbance > DISTURBANCE_HIGH
    failed = delta <= 0
    if high and failed:
        return "aware_gap"
    if high and not failed:
        return "productive_struggle"
    if not high and failed:
        return "blind_spot"
    return "mastery"

def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:
        return None
    return round(num / (dx * dy), 3)

def fuse(
    session_id: str,
    analyses: list[ChunkAnalysis],
    scores: list[Score],
    per_question: list[QuestionDelta],
) -> FusionResult:
    seg_delta = _segment_deltas(scores, per_question)
    conf_by_id = {a.chunk_id: a for a in analyses}
    ids = sorted(set(conf_by_id) | set(seg_delta))

    per_segment: list[SegmentFusion] = []
    xs: list[float] = []
    ys: list[float] = []
    for sid in ids:
        a = conf_by_id.get(sid)
        disturbance = round(1.0 - a.confidence, 3) if a else 0.0
        delta = seg_delta.get(sid)
        per_segment.append(SegmentFusion(
            segment_id=sid,
            text=a.text if a else "",
            disturbance=disturbance,
            transfer_delta=delta,
            quadrant=_quadrant(disturbance, delta),
        ))
        if a is not None and delta is not None:
            xs.append(disturbance)
            ys.append(-delta)

    counts: dict[str, int] = {}
    for s in per_segment:
        counts[s.quadrant] = counts.get(s.quadrant, 0) + 1

    return FusionResult(
        session_id=session_id,
        per_segment=per_segment,
        quadrant_counts=counts,
        calibration_rho=_pearson(xs, ys),
    )
