# ============================================================
# Multi-Source Feedback Intelligence System
# Dockerfile — production build
# ============================================================

FROM python:3.10-slim AS base

# Metadata
LABEL maintainer="your-team@company.com"
LABEL description="Multi-Source Feedback Intelligence System"

# Prevent .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System dependencies (required for torch, Pillow, reportlab)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libssl-dev \
    libpng-dev \
    libjpeg-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Dependency layer (separate from app code for caching)
# ---------------------------------------------------------------------------
FROM base AS builder

WORKDIR /build
COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install --prefix=/install -r requirements.txt

# ---------------------------------------------------------------------------
# Runtime image
# ---------------------------------------------------------------------------
FROM base AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY . .

# Create data directories
RUN mkdir -p data reports .cache

# Pre-download the HuggingFace model at build time
RUN python -c "\
from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
model_name = 'distilbert-base-uncased-finetuned-sst-2-english'; \
AutoTokenizer.from_pretrained(model_name); \
AutoModelForSequenceClassification.from_pretrained(model_name); \
print('Model downloaded successfully.')"

# Non-root user for security
RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

# Expose Streamlit default port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Streamlit configuration via env vars
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]