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
    SECRET_KEY: str = ""  # Set via environment variable
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS settings
    CORS_ORIGINS: List[str] = ["*"]
    
    # Database settings
    DATABASE_URL: str = ""  # Set via environment variable
    
    # Redis settings
    REDIS_URL: str = ""  # Set via environment variable
    
    # E2B settings
    E2B_API_KEY: str = ""  # Set via environment variable
    
    # Agent settings
    MAX_AGENTS: int = 10
    AGENT_MEMORY_LIMIT: int = 1024  # MB
    TASK_TIMEOUT: int = 300  # seconds
    
    # OpenTelemetry settings
    OTEL_SERVICE_NAME: str = "clay-ai"
    OTEL_EXPORTER_OTLP_ENDPOINT: str = "http://localhost:4317"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True
    )


settings = Settings() 