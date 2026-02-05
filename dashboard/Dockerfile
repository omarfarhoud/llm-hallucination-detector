# Stage 1: Builder
# Using the python-variant of uv to ensure a shell (/bin/sh) is available
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

# Optimized for Docker builds
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=600

WORKDIR /app

# Install dependencies using the root project files
# We use the 'dashboard' extra to keep this image lightweight
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen --no-install-project --no-dev --extra dashboard

# Stage 2: Runtime
FROM python:3.11-slim-bookworm

WORKDIR /app

# Copy the virtual environment from the builder
COPY --from=builder /app/.venv /app/.venv

# Ensure the app uses the virtual environment's python and libraries
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

# Copy the dashboard source code
# Build context must be the project root for this to work
COPY dashboard/ /app/

EXPOSE 8501

# Healthcheck to ensure Streamlit is responding
HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]