"""Deterministic confusion-engine output for the targeted-question demo — stands in for the
future ML (Divay.MD §6). A learner explaining a web request flow: most chunks are clear, a few
are low-confidence with typed anomalies. Keeps the harness output stable.
"""
from app.schemas import Anomaly, ChunkAnalysis

DEMO_CHUNKS = [
    ChunkAnalysis(chunk_id=0, text="The client sends an HTTP request to the load balancer.",
                  confidence=0.90),
    ChunkAnalysis(chunk_id=1, text="The load balancer forwards it to one of the web servers.",
                  confidence=0.86),
    ChunkAnalysis(
        chunk_id=2,
        text="Then the request goes to the, um, the mainframe I think?",
        confidence=0.25,
        anomalies=[
            Anomaly(type="factual_error", source="Space C (text/knowledge)", score=0.91,
                    evidence="the architecture uses application servers and a database, not a mainframe"),
            Anomaly(type="recall_failure", source="Space A (audio/text)", score=0.7,
                    evidence="filler 'um' + trailing rising pitch on 'mainframe?'"),
        ],
    ),
    ChunkAnalysis(chunk_id=3, text="The server queries the database to get the user data.",
                  confidence=0.88),
    ChunkAnalysis(
        chunk_id=4,
        text="And caching is handled by, maybe Redis? or the database itself, I'm not sure.",
        confidence=0.30,
        anomalies=[
            Anomaly(type="hedging", source="Space B (text/text)", score=0.8,
                    evidence="'maybe', 'not sure' + self-correction between two options"),
        ],
    ),
    ChunkAnalysis(
        chunk_id=5,
        text="The session token is stored in a cookie, or a header, one of those.",
        confidence=0.42,
        anomalies=[
            Anomaly(type="hedging", source="Space B (text/text)", score=0.6,
                    evidence="'one of those' — vague reference"),
        ],
    ),
    ChunkAnalysis(chunk_id=6, text="Finally the response is sent back to the client.",
                  confidence=0.92),
]
