FROM python:3.11-slim-bookworm

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# Use this directory as build context: docker build -t acd-gpt ./st
COPY Aptfile /tmp/Aptfile
RUN apt-get update \
 && xargs -r apt-get install -y --no-install-recommends < /tmp/Aptfile \
 && rm -rf /var/lib/apt/lists/*

# Bundled updated CLI wheel; upload this file alongside the Dockerfile.
COPY anyonecandock-1.3.1-py3-none-any.whl /opt/anyonecandock-1.3.1-py3-none-any.whl
COPY requirements-api.txt /app/requirements-api.txt
RUN python -m pip install --upgrade pip \
 && python -m pip install "/opt/anyonecandock-1.3.1-py3-none-any.whl[diagram]" -r /app/requirements-api.txt

COPY api.py /app/api.py
# JOBS and the quota lock are process-local: run one worker.
CMD ["sh", "-c", "exec uvicorn api:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
