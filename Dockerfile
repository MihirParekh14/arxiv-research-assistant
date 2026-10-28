FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# Mount data at runtime:
#   docker run --rm -v "$PWD/data:/app/data" arxiv-assistant --limit 1000
ENTRYPOINT ["python", "-m", "src.ingest"]
