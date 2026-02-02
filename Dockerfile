# ============================================================================
# Stage 1: Builder - Install dependencies
# ============================================================================
FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./

RUN uv sync --frozen --no-dev

# ============================================================================
# Stage 2: Runtime - Minimal production image
# ============================================================================
FROM python:3.11-slim

RUN useradd -m -u 1000 appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

WORKDIR /app

# Set environment variables early
# Adding /app to PYTHONPATH helps Python find the 'api' package
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy only the virtual environment
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

# Copy application code (Do this last to maximize cache usage)
COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8000

# Health check - (Ensure you actually have a /health endpoint in main.py!)
HEALTHCHECK --interval=30s \
    --timeout=5s \
    --start-period=10s \
    --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)"

# Corrected module path: api.main
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]