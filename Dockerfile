# syntax=docker/dockerfile:1
# ---------------------------------------------------------------
# Stage 1 — builder : compile/installe les dépendances dans un venv
# (pip, cache et fichiers de build restent dans ce stage)
# ---------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements-prod.txt .
RUN pip install -r requirements-prod.txt \
    # Allège le venv : tests, caches et headers inutiles au runtime
    && find /opt/venv -type d -name "__pycache__" -prune -exec rm -rf {} + \
    && find /opt/venv -type d -name "tests" -path "*/torch/*" -prune -exec rm -rf {} + \
    && rm -rf /opt/venv/lib/python3.11/site-packages/torch/include

# ---------------------------------------------------------------
# Stage 2 — runtime : uniquement le venv + le code, utilisateur non-root
# Le modèle n'est PAS dans l'image : il est téléchargé au 1er démarrage
# dans le volume hf-cache (monté sur /models).
# ---------------------------------------------------------------
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="inferloop-api" \
      org.opencontainers.image.description="API d'inférence MédiaLens (FastAPI + HuggingFace)" \
      org.opencontainers.image.source="https://github.com/DLuffy07/inferloop"

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    FEEDBACK_PATH=/data/feedback.jsonl \
    TOKENIZERS_PARALLELISM=false

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /models /data \
    && chown -R app:app /models /data

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=app:app api ./api
COPY --chown=app:app model/drift_baseline.json ./model/drift_baseline.json

USER app
EXPOSE 8000

# Pas de curl dans l'image slim : healthcheck en Python pur.
HEALTHCHECK --interval=15s --timeout=5s --start-period=300s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
