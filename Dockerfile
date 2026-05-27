FROM python:3.12-slim

# Build dependencies: gcc/g++ for compiled wheels; curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch with CUDA 12.4 support BEFORE other deps so sentence-transformers
# doesn't pull in the default CPU-only or wrong-CUDA wheel.
RUN pip install --no-cache-dir \
    torch==2.5.1+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# Install remaining Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source, ingestion scripts, and seed data
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY data/ ./data/

# HuggingFace cache — overridden by the mounted volume at runtime
ENV HF_HOME=/app/.cache/huggingface

# Run as non-root to limit blast radius if the process is compromised
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser \
    && chown -R appuser:appgroup /app \
    && mkdir -p /data/chroma && chown -R appuser:appgroup /data
USER appuser

EXPOSE 8000

# Single worker: services use module-level singletons that are not fork-safe
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
