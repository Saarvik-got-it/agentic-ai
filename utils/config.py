"""
Configuration management using Pydantic Settings.
Loads environment variables from .env file and provides strongly-typed access.
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field
import os


class Settings(BaseSettings):
    """
    Application configuration loaded from environment variables.
    Uses .env file for local development.
    """
    
    # API Configuration
    google_api_key: str = Field(..., alias="GOOGLE_API_KEY")
    
    # Paths
    document_folder: str = Field(default="./data/documents", alias="DOCUMENT_FOLDER")
    vector_store_path: str = Field(default="./vector_store/faiss_index", alias="VECTOR_STORE_PATH")
    
    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_file: str = Field(default="./logs/rag_agent.log", alias="LOG_FILE")
    
    # Retrieval Settings
    top_k_chunks: int = Field(default=5, alias="TOP_K_CHUNKS")
    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")
    
    # LLM Settings
    llm_temperature: float = Field(default=0.3, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=512, alias="LLM_MAX_TOKENS")
    
    # Email Configuration (optional, used by Email Agent)
    email_host: str = Field(default="smtp.gmail.com", alias="EMAIL_HOST")
    email_port: int = Field(default=587, alias="EMAIL_PORT")
    email_user: str = Field(default="", alias="EMAIL_USER")
    email_password: str = Field(default="", alias="EMAIL_PASSWORD")
    email_use_tls: bool = Field(default=True, alias="EMAIL_USE_TLS")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"
    
    def ensure_paths_exist(self):
        """Create necessary directories if they don't exist."""
        Path(self.document_folder).mkdir(parents=True, exist_ok=True)
        Path(self.vector_store_path).mkdir(parents=True, exist_ok=True)
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    """
    Load and return configuration settings.
    
    Returns:
        Settings instance with all configurations
    
    Raises:
        ValueError: If GOOGLE_API_KEY is not set
    """
    try:
        settings = Settings()
        settings.ensure_paths_exist()
        return settings
    except ValueError as e:
        raise ValueError(
            f"Configuration error: {e}. "
            "Ensure .env file exists with GOOGLE_API_KEY set."
        )
