from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    student_base_url: str = "https://api.openai.com/v1"
    student_api_key: str = ""
    student_model: str = "gpt-4o-mini"

    generator_base_url: str = "https://api.openai.com/v1"
    generator_api_key: str = ""
    generator_model: str = "gpt-4o"

    max_concurrency: int = 40

    n_taught: int = 3
    n_cold: int = 2

    question_confidence_threshold: float = 0.5

    question_gate_on_anomalies: bool = False

    filter_cold_samples: int = 2
    n_candidate_questions: int = 8

    verifier_model: str = ""

    ml_service_url: str = "http://localhost:8100"
    ml_service_timeout: float = 60.0
    ml_service_health_timeout: float = 20.0

    store_backend: str = "memory"
    database_url: str = ""

    cors_origins: str = "http://127.0.0.1:5173,http://localhost:5173"

    default_classes: int = 5
    notes_concurrency: int = 4
    notes_timeout: float = 60.0

    objective_check_every: int = 1
    goal_probe_cooldown: int = 2

settings = Settings()
