# syntax=docker/dockerfile:1

# ---- Build stage: install Python deps into an isolated user site-packages dir ----
FROM python:3.11-slim AS builder

WORKDIR /build

# build-essential is needed to build a couple of sentence-transformers'
# transitive dependencies from source on some platforms; removed again in
# the runtime stage so it doesn't bloat the final image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ---- Runtime stage: slim image, no build tools, non-root user ----
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/home/appuser/.local/bin:$PATH \
    HF_HOME=/home/appuser/.cache/huggingface

RUN useradd --create-home --uid 1000 appuser

COPY --from=builder /root/.local /home/appuser/.local
COPY app ./app
COPY sample_docs ./sample_docs
COPY scripts ./scripts

RUN chown -R appuser:appuser /app /home/appuser/.local
USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8080/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
