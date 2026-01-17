"""Application settings and configuration."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Model Configuration
    model_name: str = "phi3:mini"
    ollama_host: str = "http://localhost:11434"
    
    # Judge Threshold (0-1 scale)
    judge_threshold: float = 0.6
    
    # Citation Threshold 
    citation_threshold: float = 0.7
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Singleton instance
settings = Settings()