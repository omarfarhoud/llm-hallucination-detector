"""
FastAPI application for LLM hallucination detection.

Provides REST endpoints for detecting hallucinations in LLM outputs
using multi-signal detection (similarity, judge, citation).
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import health, detect

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


# ============================================================================
# Lifespan Event Handler
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for startup and shutdown.
    
    Replaces deprecated on_event decorators.
    """
    # Startup
    logger.info("🚀 LLM Hallucination Detector API starting up...")
    yield
    # Shutdown
    logger.info("👋 LLM Hallucination Detector API shutting down...")


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="LLM Hallucination Detector",
    description=(
        "Multi-signal hallucination detection system for LLM outputs. "
        "Combines semantic similarity, LLM-as-judge, and citation verification "
        "to detect when answers deviate from provided context."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ============================================================================
# CORS Middleware (for frontend access)
# ============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Include Routers
# ============================================================================

app.include_router(health.router, tags=["Health"])
app.include_router(detect.router, tags=["Detection"])

# ============================================================================
# Root Endpoint
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "LLM Hallucination Detector",
        "version": "0.1.0",
        "endpoints": {
            "health": "/health",
            "detect": "/detect",
            "docs": "/docs",
            "redoc": "/redoc"
        },
        "description": "Multi-signal hallucination detection for LLM outputs"
    }