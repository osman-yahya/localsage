FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/hf_cache \
    SENTENCE_TRANSFORMERS_HOME=/data/hf_cache

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src ./src
COPY config ./config
COPY scripts ./scripts

ENV PYTHONPATH=/app

# CMD (not ENTRYPOINT) so `docker compose run --rm app python -m scripts.bootstrap`
# can replace the default. Without args (./localsage), the CLI starts.
CMD ["python", "-u", "-m", "src.main"]
