# Use Python 3.12 slim as base image
FROM python:3.12-slim-bookworm AS builder

# Install uv with version pinning for reproducibility
COPY --from=ghcr.io/astral-sh/uv:0.6.1 /uv /uvx /bin/

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create and activate virtual environment
ENV VIRTUAL_ENV=/app/.venv
RUN uv venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Set up uv configuration for optimal Docker usage
ENV UV_SYSTEM_PYTHON=0
ENV UV_CACHE_DIR=/root/.cache/uv
ENV UV_LINK_MODE=copy
ENV UV_COMPILE_BYTECODE=1

# Install dependencies in separate layers for better caching
# First, copy and install from requirements.txt
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r requirements.txt

# Create final slim image
FROM python:3.12-slim-bookworm

# Copy uv from builder stage
COPY --from=ghcr.io/astral-sh/uv:0.6.1 /uv /uvx /bin/

# Create non-root user
RUN useradd --create-home appuser

# Copy virtual environment from builder
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=0 \
    UV_CACHE_DIR=/root/.cache/uv \
    UV_LINK_MODE=copy

# Create and set permissions for tmp directory
RUN mkdir -p /tmp/uploads && chown appuser:appuser /tmp/uploads

# Switch to non-root user
USER appuser

# Set working directory
WORKDIR /app

# Copy only necessary application files
COPY --chown=appuser:appuser ./main.py /app/
COPY --chown=appuser:appuser ./schema /app/schema/

# Create models directory and set permissions
RUN mkdir -p /app/models && chown appuser:appuser /app/models

# Switch back to root temporarily to install additional dependencies
USER root

# Install additional dependencies needed for document processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Switch back to appuser
USER appuser

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
