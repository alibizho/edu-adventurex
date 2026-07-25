"""Environment-based settings. See .env.example."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Student ensemble — small/fast/cheap keeps the transfer delta sensitive.
    student_base_url: str = "https://api.openai.com/v1"
    student_api_key: str = ""
    student_model: str = "gpt-4o-mini"

    # Question generation + attribution — stronger model.
    generator_base_url: str = "https://api.openai.com/v1"
    generator_api_key: str = ""
    generator_model: str = "gpt-4o"

    # Ensemble fan-out concurrency.
    max_concurrency: int = 40

    # Ensemble sizes. 3 taught + 2 cold = 5 total (fast demo mode; the delta still has a
    # 2-persona cold control). Scale up for a real measurement run.
    n_taught: int = 3
    n_cold: int = 2

    # Real-time question gate: emit a question only when a chunk is confused — has an anomaly
    # OR confidence below this threshold. Aligns with fusion.py DISTURBANCE_HIGH = 0.5.
    question_confidence_threshold: float = 0.5

    # Whether the real-time gate also fires on ml-service anomalies (logic_error/recall_failure).
    # Default off: the on-box judges are noisy on short utterances (Space B flags non-contradictions,
    # Space A misses hedging), so firing on anomalies yields false positives. Flip on once the
    # ml-service is retuned. Space C factual_errors are suppressed via enable_space_c=False.
    question_gate_on_anomalies: bool = False

    # /measure cost knobs. The cold-student filter (not the scoring ensemble) dominates /measure
    # time, so these are the real speed levers. filter_cold_samples = cold samples per candidate
    # question; n_candidate_questions = how many the generator writes.
    filter_cold_samples: int = 2
    n_candidate_questions: int = 8

    # Grader verifier model; falls back to generator_model when empty.
    verifier_model: str = ""

    # --- Confusion analysis (Instrument B/C) lives in the ml-service; the backend forwards audio
    #     to it. On failure the client degrades to a neutral analysis so a demo never hard-fails.
    #     The text-only /confusion/mock endpoint stays for offline dev without the GPU box. ---
    ml_service_url: str = "http://localhost:8100"
    ml_service_timeout: float = 60.0            # first call includes cold Whisper on the GPU box
    # Health probe budget. Separate from the timeout above (that one covers real inference) but NOT
    # a token value: the box is reachable over a tunnel to a rented GPU, and a bare /health round
    # trip measures 4-8s from here. At the old hardcoded 5s this probe timed out intermittently and
    # the class fell back to typing while the GPU was up and healthy.
    ml_service_health_timeout: float = 20.0

    # --- Context store: "memory" (dev default, lost on restart, no extra deps) or "db" (Postgres,
    #     durable across restarts). When store_backend == "db", database_url must be set. ---
    store_backend: str = "memory"
    database_url: str = ""

    # Browser origins for the separately hosted Vite frontend. Parsed in app.main.
    cors_origins: str = "http://127.0.0.1:5173,http://localhost:5173"

    # --- Learning plan (curriculum). Default number of classes when scope/build don't decide one. ---
    default_classes: int = 5

    # --- Objective mastery. How many new utterances to accumulate before judging which class
    #     objectives they covered: one verifier call per check, so this is the cost/latency dial.
    #     1 = judge after every chunk. Batching at 3 was the visible lag in the CLASS GOALS panel:
    #     chunks are pause-delimited and uploaded serially, so three of them is 15-30s of talking
    #     before a checkmark could even be considered. The call runs in a background task the
    #     teaching turn never waits on, so the cost is tokens, not latency. Raise it to trade
    #     freshness back for spend. ---
    objective_check_every: int = 1
    # Turns to wait before a student may again nudge the learner toward an uncovered objective.
    # This is what stops a clear utterance producing no reaction at all, so it is short: the room
    # steers roughly every other turn. Raise it if the class feels naggy, lower it to 1 to have a
    # student respond to almost everything.
    goal_probe_cooldown: int = 2


settings = Settings()
