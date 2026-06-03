# Stage 1: Build Stage
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the uv binary from the official image
COPY --from=astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV UV_PYTHON=/usr/bin/python3.10
ENV UV_COMPILE_BYTECODE=1

# Copy dependency configs first to maximize Docker layer caching
COPY pyproject.toml uv.lock ./

# --no-install-project because code hasn't been copied yet
RUN uv venv .venv && \
    uv sync --frozen --no-install-project

# Stage 2: Production Stage
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS runner

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the pre built venv from the builder stage
COPY --from=builder /app/.venv /app/.venv

COPY . .

ENV PATH="/app/.venv/bin:$PATH"

CMD ["python", "main.py"]

