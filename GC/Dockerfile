FROM python:3.11-slim

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── System packages ───────────────────────────────────────────────────────────
# Includes Open Babel and rendering libs needed by RDKit/CairoSVG/Streamlit app.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    openbabel \
    libopenbabel-dev \
    libcairo2-dev \
    libpango1.0-dev \
    libpangocairo-1.0-0 \
    pkg-config \
    python3-dev \
    libgirepository1.0-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Python packages ───────────────────────────────────────────────────────────
# Install from requirements only to avoid duplicate installs.
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── AutoDock Vina ─────────────────────────────────────────────────────────────
# Preinstall Vina in the image so Cloud Run instances do not need to download it
# at runtime during cold start.
RUN curl -L \
    https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.7/vina_1.2.7_linux_x86_64 \
    -o /usr/local/bin/vina && \
    chmod +x /usr/local/bin/vina

# ── App files ─────────────────────────────────────────────────────────────────
COPY . .

# ── Streamlit / Cloud Run runtime ─────────────────────────────────────────────
# Cloud Run injects PORT dynamically. Streamlit must bind to 0.0.0.0.
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8080

CMD ["sh", "-c", "streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-8080} --server.headless=true --server.fileWatcherType=none --browser.gatherUsageStats=false"]
