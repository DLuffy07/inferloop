FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements-prod.txt .

RUN pip install \
    --no-cache-dir \
    --prefix=/install \
    -r requirements-prod.txt


FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /install /usr/local
COPY api ./api

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]