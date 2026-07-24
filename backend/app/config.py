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

    # POC ensemble sizes (scale to 20 / 10 for the full run; §4.4).
    n_taught: int = 8
    n_cold: int = 4

    # Grader verifier model; falls back to generator_model when empty.
    verifier_model: str = ""


settings = Settings()
