FROM python:3.12-slim

WORKDIR /app

# System dependencies for sentence-transformers and chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy everything needed to build the package
COPY pyproject.toml .
COPY src/ src/

# Install the package and dependencies
RUN pip install --no-cache-dir ".[dev]" || pip install --no-cache-dir .

# Copy the FastAPI wrapper
COPY api.py .

# Volume for persistent ChromaDB storage
VOLUME ["/data"]

EXPOSE 8001

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]
