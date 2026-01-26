"""
Pydantic models for API request/response validation.

These models define the contract between the API and clients,
ensuring type safety and automatic validation.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional


# ============================================================================
# Context Document Schema
# ============================================================================

class ContextDocument(BaseModel):
    """
    A single context document with ID and text.
    
    Used to structure retrieved/provided context for citation verification.
    """
    id: int = Field(..., ge=1, description="Document ID (1-indexed)")
    text: str = Field(..., min_length=1, description="Document text content")


# ============================================================================
# Request Schemas
# ============================================================================

class DetectionRequest(BaseModel):
    """
    Request for hallucination detection.
    
    Contains question, structured context, and answer to evaluate.
    """
    question: str = Field(
        ...,
        min_length=1,
        description="The question being answered"
    )
    context: List[ContextDocument] = Field(
        ...,
        min_items=1,
        description="List of context documents (what would be retrieved in RAG)"
    )
    answer: str = Field(
        ...,
        min_length=1,
        description="The answer to evaluate for hallucinations"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are the risk factors for type 2 diabetes?",
                "context": [
                    {
                        "id": 1,
                        "text": "Type 2 diabetes is characterized by insulin resistance."
                    },
                    {
                        "id": 2,
                        "text": "Risk factors include obesity, physical inactivity, and family history."
                    }
                ],
                "answer": "Risk factors include obesity, lack of exercise, and genetics [2]."
            }
        }


# ============================================================================
# Response Schemas
# ============================================================================

class SignalScore(BaseModel):
    """Individual signal's hallucination score."""
    hallucination_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Hallucination score (0=faithful, 1=hallucinated)"
    )
    raw: Dict = Field(..., description="Raw signal output")


class AggregationMode(BaseModel):
    """Metadata about how signals were aggregated."""
    strategy: str = Field(
        ...,
        description="Aggregation strategy used (full_weighted, structural_gate, judge_dominant)"
    )
    used_signals: List[str] = Field(
        ...,
        description="List of signals used in aggregation"
    )


class DetectionResponse(BaseModel):
    """
    Response from hallucination detection.
    
    Contains final verdict, scores, and detailed signal breakdowns.
    """
    hallucination_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Final aggregated hallucination score"
    )
    is_hallucination: bool = Field(
        ...,
        description="Binary decision: true if hallucination detected"
    )
    threshold: float = Field(
        ...,
        description="Threshold used for binary decision"
    )
    weights: Dict[str, float] = Field(
        ...,
        description="Signal weights used in aggregation"
    )
    aggregation_mode: AggregationMode = Field(
        ...,
        description="Metadata about aggregation strategy"
    )
    signal_scores: Dict[str, float] = Field(
        ...,
        description="Individual signal hallucination scores"
    )
    signals: Dict[str, SignalScore] = Field(
        ...,
        description="Detailed signal outputs"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "hallucination_score": 0.23,
                "is_hallucination": False,
                "threshold": 0.7,
                "weights": {
                    "semantic_similarity": 0.25,
                    "llm_judge": 0.60,
                    "citation_check": 0.15
                },
                "aggregation_mode": {
                    "strategy": "full_weighted",
                    "used_signals": ["semantic_similarity", "llm_judge", "citation_check"]
                },
                "signal_scores": {
                    "semantic_similarity_h": 0.15,
                    "llm_judge_h": 0.25,
                    "citation_check_h": 0.0
                },
                "signals": {
                    "semantic_similarity": {
                        "hallucination_score": 0.15,
                        "raw": {}
                    },
                    "llm_judge": {
                        "hallucination_score": 0.25,
                        "raw": {}
                    },
                    "citation_check": {
                        "hallucination_score": 0.0,
                        "raw": {}
                    }
                }
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    model_name: str = Field(..., description="LLM model in use")
    ollama_connected: bool = Field(..., description="Ollama connection status")