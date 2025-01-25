"""Configuration settings for Clay AI using Pydantic V2."""

from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings class for Clay AI application."""

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    LOG_LEVEL: str = "info"

    # Security settings
    SECRET_KEY: str = "your-secret-key-here"  # Change in production
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days

    # CORS settings
    CORS_ORIGINS: List[str] = ["*"]

    # Database settings
    DATABASE_URL: str = (
        "mysql+aiomysql://clay_user:clay_password@localhost:3306/clay_ai"
    )
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 10

    # Redis settings
    REDIS_URL: str = ""  # Set via environment variable

    # E2B settings
    E2B_API_KEY: str = ""  # Set your E2B API key here

    # Agent settings
    MAX_AGENTS: int = 10
    AGENT_MEMORY_LIMIT: int = 1024  # MB
    TASK_TIMEOUT: int = 300  # seconds

    # OpenTelemetry settings
    OTEL_SERVICE_NAME: str = "clay-ai"
    OTEL_EXPORTER_OTLP_ENDPOINT: str = "http://localhost:4317"

    # API settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Clay AI"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=True
    )


settings = Settings()
