# ============================================================================
# Stage 1: Builder - Install dependencies
# ============================================================================
FROM python:3.11-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files AND README (needed for build)
COPY pyproject.toml uv.lock README.md ./

# ✅ UPDATE: Added --extra api and --no-install-project
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_HTTP_TIMEOUT=600 uv sync --frozen --no-dev --extra api --no-install-project

# Pre-download the sentence-transformers embedding model so it is baked into
# the image. Without this it is fetched from HuggingFace on every cold start
# (~40s delay + a runtime network dependency). Must match the model name in
# src/detection/similarity_check.py.
ENV HF_HOME=/opt/hf
RUN /app/.venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# ============================================================================
# Stage 2: Runtime - Minimal production image
# ============================================================================
FROM python:3.11-slim

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

WORKDIR /app

# Set environment variables
# HF_HOME points at the pre-baked model cache; HF_HUB_OFFLINE=1 keeps the
# container from reaching out to HuggingFace at runtime (instant model load).
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1

# Copy only the virtual environment from builder
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

# Copy the pre-downloaded embedding model cache from builder
COPY --from=builder --chown=appuser:appuser /opt/hf /opt/hf

# Copy application code last to maximize layer caching
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s \
    --timeout=5s \
    --start-period=10s \
    --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)"

# Command to run the backend
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]