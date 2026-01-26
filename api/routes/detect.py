"""Hallucination detection endpoint."""

import logging
import asyncio
from fastapi import APIRouter, HTTPException

from src.models.schemas import DetectionRequest, DetectionResponse
from src.detection.similarity_check import SimilarityChecker
from src.detection.llm_judge import llm_judge
from src.detection.citation_check import citation_checker
from src.detection.aggregator import aggregator

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize outside the route to act as a singleton (prevents reloading models)
similarity_checker = SimilarityChecker()

@router.post("/detect", response_model=DetectionResponse)
async def detect_hallucination(request: DetectionRequest):
    """
    Detect hallucinations in an LLM answer using parallel signal processing.
    """
    try:
        logger.info(f"Processing detection request for question: {request.question[:50]}...")
        
        # Prepare context data
        combined_context = "\n".join([f"Document {doc.id}: {doc.text}" for doc in request.context])
        context_docs = [{"doc_id": doc.id, "text": doc.text} for doc in request.context]
        
        # ----------------------------------------------------------------
        # Run detection signals IN PARALLEL
        # ----------------------------------------------------------------
        logger.debug("Dispatching detection signals...")
        
        # asyncio.gather allows the LLM Judge (I/O bound) and 
        # Similarity (CPU/Network bound) to run at the same time.
        similarity_task = asyncio.to_thread(similarity_checker.check, request.answer, context_docs)
        judge_task = llm_judge.judge(request.question, request.answer, combined_context)
        citation_task = asyncio.to_thread(citation_checker.check, request.answer, context_docs)

        similarity_result, judge_result, citation_result = await asyncio.gather(
            similarity_task, 
            judge_task, 
            citation_task
        )
        
        # ----------------------------------------------------------------
        # Aggregate signals
        # ----------------------------------------------------------------
        logger.debug("Aggregating signals...")
        result = aggregator.aggregate(
            similarity_result=similarity_result,
            judge_result=judge_result,
            citation_result=citation_result
        )
        
        logger.info(
            f"Detection complete: score={result['hallucination_score']:.3f}, "
            f"verdict={'HALLUCINATION' if result['is_hallucination'] else 'FAITHFUL'}"
        )
        
        return DetectionResponse(**result)
        
    except asyncio.TimeoutError:
        logger.error("Detection timed out (likely LLM provider)")
        raise HTTPException(status_code=504, detail="Detection timed out")
    except Exception as e:
        logger.error(f"Detection failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal detection error: {str(e)}"
        )