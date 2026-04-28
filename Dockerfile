FROM python:3.12-slim

# Build dependencies: gcc/g++ for compiled wheels; curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies before copying app code (layer-cache friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source, ingestion scripts, and seed data
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY data/ ./data/

# HuggingFace cache — overridden by the mounted volume at runtime
ENV HF_HOME=/root/.cache/huggingface

EXPOSE 8000

# Single worker: services use lru_cache singletons that are not fork-safe
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
