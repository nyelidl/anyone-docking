# ══════════════════════════════════════════════════════════════════════════════
#  Dockerfile — Anyone Can Dock
#  Optimised for Google Cloud Run deployment.
#
#  Build:  docker build -t anyone-can-dock .
#  Run:    docker run -p 8080:8080 anyone-can-dock
# ══════════════════════════════════════════════════════════════════════════════

FROM python:3.11-slim

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── System packages ───────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
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
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── AutoDock Vina 1.2.7 ───────────────────────────────────────────────────────
# Pre-install at build time — eliminates cold-start GitHub download (~15 MB).
# core.py's get_vina_binary() finds this via shutil.which("vina") immediately.
RUN curl -fsSL \
    https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.7/vina_1.2.7_linux_x86_64 \
    -o /usr/local/bin/vina && \
    chmod +x /usr/local/bin/vina && \
    vina --version

# ── App source files ──────────────────────────────────────────────────────────
COPY . .

# Verify .streamlit/config.toml made it into the image.
# Fails the build early if config.toml is missing (e.g. gitignored),
# which is much easier to debug than a silent Safari runtime error.
RUN test -f /app/.streamlit/config.toml && \
    echo "✓ .streamlit/config.toml present" || \
    (echo "ERROR: .streamlit/config.toml missing — add it to git and rebuild" && exit 1)

# ── Environment variables ─────────────────────────────────────────────────────
# Mirror .streamlit/config.toml as ENV fallbacks.
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
ENV STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
# Safari fix: allowedOrigins set via config.toml (not CLI) to avoid
# shell quoting issues with "*" that crash Streamlit 1.40+ at startup.
ENV STREAMLIT_SERVER_ALLOWED_ORIGINS="*.run.app"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

# ── Entrypoint ────────────────────────────────────────────────────────────────
# allowedOrigins is intentionally NOT passed as a CLI flag —
# the bare "*" argument causes Streamlit to crash before binding the port.
# It is set in .streamlit/config.toml and the ENV var above instead.
CMD ["sh", "-c", \
     "streamlit run app.py \
      --server.address=0.0.0.0 \
      --server.port=${PORT:-8080} \
      --server.headless=true \
      --server.enableCORS=false \
      --server.enableXsrfProtection=false \
      --server.fileWatcherType=none \
      --browser.gatherUsageStats=false"]
