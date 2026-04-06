# ══════════════════════════════════════════════════════════════════════════════
#  Anyone Can Dock — Google Cloud Run Dockerfile
#  Optimised for:
#    • Zero cold-start Vina download (binary baked into image)
#    • Safari compatibility (CORS + XSRF + allowedOrigins)
#    • Fast layer-cached rebuilds (deps before code)
#    • Minimal image size (--no-install-recommends, cache purged)
# ══════════════════════════════════════════════════════════════════════════════

FROM python:3.11-slim

# ── Working directory ──────────────────────────────────────────────────────────
WORKDIR /app

# ── System packages ────────────────────────────────────────────────────────────
# Pinned in one RUN layer so apt cache is never left in the image.
# build-essential  → needed to compile some Python C extensions at pip install
# curl             → used to download Vina binary below
# openbabel        → molecular format conversion (core.py run_cmd calls obabel)
# libopenbabel-dev → header files for any openbabel Python bindings
# libcairo2-dev    → CairoSVG (svg_to_png in core.py)
# libpango1.0-dev  → text rendering for CairoSVG
# libpangocairo-1.0-0 → Pango+Cairo bridge
# pkg-config       → required by some pip packages that probe for cairo/pango
# python3-dev      → C extension headers
# libgirepository1.0-dev → GObject introspection (cairosvg dependency chain)
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

# ── Python packages ────────────────────────────────────────────────────────────
# Copy requirements first — Docker caches this layer separately so code edits
# to app.py / core.py do not trigger a full pip reinstall.
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── AutoDock Vina 1.2.7 (baked into image) ────────────────────────────────────
# Downloading at build time means Cloud Run cold starts have zero network
# dependency for Vina.  get_vina_binary() in core.py resolves this via
# shutil.which("vina") at step 1, never reaching the runtime download path.
RUN curl -fsSL \
    https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.7/vina_1.2.7_linux_x86_64 \
    -o /usr/local/bin/vina \
 && chmod +x /usr/local/bin/vina \
 && vina --version   # smoke-test — fails the build if the binary is bad

# ── Streamlit config ───────────────────────────────────────────────────────────
# Copied before app code so this layer is cached independently.
COPY .streamlit/ .streamlit/

# ── Application code ───────────────────────────────────────────────────────────
COPY app.py core.py ./

# ── Runtime environment ────────────────────────────────────────────────────────
# These ENV vars are belt-and-suspenders alongside .streamlit/config.toml.
# Cloud Run injects $PORT (usually 8080); we fall back to 8080 if not set.
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
ENV STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8080

# Use sh -c so ${PORT:-8080} is expanded by the shell at container start.
CMD ["sh", "-c", \
     "streamlit run app.py \
      --server.address=0.0.0.0 \
      --server.port=${PORT:-8080} \
      --server.headless=true \
      --server.enableCORS=false \
      --server.enableXsrfProtection=false \
      --server.fileWatcherType=none \
      --browser.gatherUsageStats=false"]
