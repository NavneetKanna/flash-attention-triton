# Stage 1: Build Stage
FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04 AS builder

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

COPY --from=astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV UV_PYTHON=/usr/bin/python3.10
ENV UV_COMPILE_BYTECODE=1
ENV UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app

# Copy dependency configs first to maximize Docker layer caching
COPY pyproject.toml uv.lock ./

# uv sync creates the env at UV_PROJECT_ENVIRONMENT (/opt/venv)
# --no-install-project because the code comes from the bind-mount at runtime
RUN uv sync --frozen --no-install-project

# Stage 2: Production Stage
FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04 AS runner

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

CMD ["python", "main.py"]
