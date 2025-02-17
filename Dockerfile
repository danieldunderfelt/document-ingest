FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Enable bytecode compilation and set copy mode for mounted volumes
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Install system dependencies required by docling
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies with caching
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r requirements.txt

# Copy application code
COPY . .

# Create tmp directory for file storage
RUN mkdir -p /tmp

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

# Reset the entrypoint
ENTRYPOINT []

# Run the FastAPI application
CMD ["python", "main.py"] 