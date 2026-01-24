"""Health check endpoint."""

from fastapi import APIRouter
from src.models.schemas import HealthResponse
from src.detection.llm_judge import llm_judge
from config.settings import settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Verifies that the API is running and can connect to Ollama.
    """
    # Test Ollama connection
    try:
        # Simple test: check if we can call the judge's LLM client
        # Without actually running a full judgment
        test_result = await llm_judge._call_llm("test")
        ollama_connected = len(test_result) > 0
    except Exception:
        ollama_connected = False
    
    status = "healthy" if ollama_connected else "degraded"
    
    return HealthResponse(
        status=status,
        model_name=settings.model_name,
        ollama_connected=ollama_connected
    )