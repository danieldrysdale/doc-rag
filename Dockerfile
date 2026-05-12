FROM python:3.12-slim

WORKDIR /app

# System dependencies for sentence-transformers and chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first for better layer caching
COPY pyproject.toml .
RUN pip install --no-cache-dir ".[dev]" || pip install --no-cache-dir .

# Copy source
COPY src/ src/

# Copy the FastAPI wrapper
COPY api.py .

# Volume for persistent ChromaDB storage
VOLUME ["/data"]

EXPOSE 8001

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]
