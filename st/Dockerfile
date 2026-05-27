FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    openbabel \
    libopenbabel-dev \
    libcairo2-dev \
    libpango1.0-dev \
    libpangocairo-1.0-0 \
    pkg-config \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install anyonecandock with all extras (includes Vina, diagram, ADMET)
RUN pip install --upgrade pip && \
    pip install "anyonecandock[all]" \
    fastapi \
    uvicorn[standard] \
    pydantic \
    requests

# Only copy api.py — no more core.py / pKaNET.py needed!
COPY api.py .

CMD uvicorn api:app --host 0.0.0.0 --port ${PORT:-10000} --workers 2
