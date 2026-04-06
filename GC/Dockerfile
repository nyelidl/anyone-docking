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
# Installed in a single layer to keep image size minimal.
# - openbabel / libopenbabel-dev : receptor & ligand file conversion
# - libcairo2-dev / libpango*    : CairoSVG rendering (2D diagrams, PNG export)
# - pkg-config / python3-dev     : compile-time headers for Python C extensions
# - libgirepository1.0-dev       : GObject introspection (pycairo dependency)
# - curl                         : download Vina binary at build time
# NOTE: build-essential and curl are removed from the final image to reduce
#       attack surface — they are only needed during the RUN steps below.
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
# Copy only requirements first so this layer is cached unless requirements change.
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── AutoDock Vina 1.2.7 ───────────────────────────────────────────────────────
# Pre-install at build time so Cloud Run cold starts have zero network
# dependency on GitHub.  Without this, every cold start downloads ~15 MB
# before the app can serve any request.
# The get_vina_binary() function in core.py checks shutil.which("vina") first
# and uses this binary immediately without any download.
RUN curl -fsSL \
    https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.7/vina_1.2.7_linux_x86_64 \
    -o /usr/local/bin/vina && \
    chmod +x /usr/local/bin/vina && \
    # Smoke-test: confirm the binary executes correctly before finalising layer
    vina --version

# ── App source files ──────────────────────────────────────────────────────────
# Copy after package installs so source edits don't bust the package cache.
COPY . .

# ── Streamlit / Cloud Run environment ────────────────────────────────────────
# These ENV vars mirror .streamlit/config.toml and act as a safe fallback.
# Streamlit reads config.toml first; ENV vars override if config.toml is absent.
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
ENV STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
# allowedOrigins: required for Safari — without this Safari blocks Streamlit's
# component ES module scripts with "TypeError: Importing a module script failed".
# Chrome is lenient about origin validation; Safari enforces it strictly.
ENV STREAMLIT_SERVER_ALLOWED_ORIGINS="*"
# Silence Python bytecode warnings in logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

# ── Entrypoint ────────────────────────────────────────────────────────────────
# Cloud Run injects $PORT at runtime (default 8080).
# All flags are explicit here so Safari and Cloud Run fixes apply even if
# .streamlit/config.toml is not present in the build context.
CMD ["sh", "-c", \
     "streamlit run app.py \
      --server.address=0.0.0.0 \
      --server.port=${PORT:-8080} \
      --server.headless=true \
      --server.enableCORS=false \
      --server.enableXsrfProtection=false \
      --server.allowedOrigins=* \
      --server.fileWatcherType=none \
      --browser.gatherUsageStats=false"]
