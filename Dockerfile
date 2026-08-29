FROM python:3.12-slim

ARG EMBEDDING_MODEL=intfloat/multilingual-e5-small

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/appuser/.cache/huggingface \
    EMBEDDING_MODEL=${EMBEDDING_MODEL}

WORKDIR /app

RUN useradd \
        --create-home \
        --uid 10001 \
        appuser \
    && mkdir -p \
        /app/data/raw \
        /app/data/chroma \
        "${HF_HOME}" \
    && chown -R appuser:appuser \
        /app \
        /home/appuser

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

USER appuser

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')"

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY --chown=appuser:appuser . .

EXPOSE 8000

HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=10s \
    --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3).read()"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
