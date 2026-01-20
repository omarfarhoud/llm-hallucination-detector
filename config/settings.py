"""Application settings and configuration."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Model Configuration
    model_name: str = "phi3:mini"
    ollama_host: str = "http://localhost:11434"
    
    # Judge Threshold (0-1 scale)
    judge_threshold: float = 0.7
    
    # Citation Threshold 
    citation_threshold: float = 0.7
    
    # -----------------------------
    # Aggregation configuration
    # -----------------------------
    similarity_weight: float = 0.25
    judge_weight: float = 0.60
    citation_weight: float = 0.15
    aggregation_threshold: float = 0.7
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Singleton instance
settings = Settings()