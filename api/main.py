import hashlib
import json
import os
import time

import redis
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field
from transformers import pipeline

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

LABELS = [
    "politique",
    "économie",
    "sport",
    "culture",
    "faits divers",
]

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))


# ============================================================
# APPLICATION FASTAPI
# ============================================================

app = FastAPI(
    title="InferLoop API",
    description="API d'inférence NLP pour MediaLens",
    version="1.0.0",
)


# ============================================================
# METRIQUES PROMETHEUS
# ============================================================

INFERENCE_COUNT = Counter(
    "inferloop_inference_total",
    "Nombre total d'inférences exécutées par le modèle",
)

INFERENCE_LATENCY = Histogram(
    "inferloop_inference_duration_seconds",
    "Temps d'inférence du modèle en secondes",
)

PREDICTION_COUNT = Counter(
    "inferloop_prediction_total",
    "Nombre de prédictions par catégorie",
    ["categorie"],
)

CACHE_HIT_COUNT = Counter(
    "inferloop_cache_hit_total",
    "Nombre de résultats récupérés depuis Redis",
)

CACHE_MISS_COUNT = Counter(
    "inferloop_cache_miss_total",
    "Nombre de résultats absents du cache Redis",
)

REQUEST_COUNT = Counter(
    "inferloop_request_total",
    "Nombre total de requêtes reçues sur /infer",
)

ERROR_COUNT = Counter(
    "inferloop_error_total",
    "Nombre total d'erreurs lors des inférences",
)


# ============================================================
# REDIS
# ============================================================

redis_client = None

try:
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        decode_responses=True,
        socket_connect_timeout=2,
        socket_timeout=2,
    )

    redis_client.ping()
    print("Connexion Redis OK.")

except Exception as error:
    print(f"Redis indisponible : {error}")
    redis_client = None


# ============================================================
# CHARGEMENT DU MODELE
# ============================================================

print("Chargement du modèle NLP...")

classifier = None

try:
    start = time.perf_counter()

    classifier = pipeline(
        "zero-shot-classification",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
    )

    loading_time = time.perf_counter() - start

    print(f"Modèle chargé en {loading_time:.2f} secondes.")

except Exception as error:
    print(f"Erreur lors du chargement du modèle : {error}")


# ============================================================
# SCHEMA D'ENTREE
# ============================================================

class InferRequest(BaseModel):
    texte: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Texte de l'article à classifier",
    )


# ============================================================
# CLE DE CACHE
# ============================================================

def get_cache_key(text: str) -> str:
    text_hash = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

    return f"inferloop:{text_hash}"


# ============================================================
# GET /health
# ============================================================

@app.get("/health")
def health():
    redis_available = False

    if redis_client is not None:
        try:
            redis_available = redis_client.ping()
        except Exception:
            redis_available = False

    return {
        "status": "healthy" if classifier is not None else "unhealthy",
        "model_loaded": classifier is not None,
        "redis_connected": bool(redis_available),
        "model": MODEL_NAME,
    }


# ============================================================
# POST /infer
# ============================================================

@app.post("/infer")
def infer(request: InferRequest):

    # Compte toutes les requêtes valides arrivant sur /infer
    REQUEST_COUNT.inc()

    # Chronomètre global de la requête
    start_request = time.perf_counter()

    if classifier is None:
        ERROR_COUNT.inc()

        raise HTTPException(
            status_code=503,
            detail="Le modèle n'est pas chargé.",
        )

    cache_key = get_cache_key(request.texte)

    # --------------------------------------------------------
    # RECHERCHE DANS REDIS
    # --------------------------------------------------------

    if redis_client is not None:
        try:
            cached_result = redis_client.get(cache_key)

            if cached_result is not None:
                CACHE_HIT_COUNT.inc()

                cached_data = json.loads(cached_result)

                # Mesure de la vraie latence de récupération Redis
                cache_latency_ms = (
                    time.perf_counter() - start_request
                ) * 1000

                cached_data["latence_ms"] = round(
                    cache_latency_ms,
                    2,
                )

                cached_data["cache"] = "HIT"

                return cached_data

            CACHE_MISS_COUNT.inc()

        except Exception as error:
            print(f"Erreur Redis GET : {error}")

    # --------------------------------------------------------
    # INFERENCE DU MODELE
    # --------------------------------------------------------

    start_inference = time.perf_counter()

    try:
        result = classifier(
            request.texte,
            candidate_labels=LABELS,
        )

    except Exception as error:
        ERROR_COUNT.inc()

        print(f"Erreur lors de l'inférence : {error}")

        raise HTTPException(
            status_code=500,
            detail="Erreur lors de l'inférence.",
        )

    inference_seconds = (
        time.perf_counter() - start_inference
    )

    predicted_category = result["labels"][0]
    confidence_score = result["scores"][0]

    # --------------------------------------------------------
    # METRIQUES PROMETHEUS
    # --------------------------------------------------------

    INFERENCE_COUNT.inc()

    INFERENCE_LATENCY.observe(
        inference_seconds
    )

    PREDICTION_COUNT.labels(
        categorie=predicted_category
    ).inc()

    # --------------------------------------------------------
    # LATENCE TOTALE DE LA REQUETE
    # --------------------------------------------------------

    total_latency_ms = (
        time.perf_counter() - start_request
    ) * 1000

    response_data = {
        "categorie": predicted_category,
        "score": round(confidence_score, 4),
        "latence_ms": round(total_latency_ms, 2),
        "cache": "MISS",
    }

    # --------------------------------------------------------
    # STOCKAGE DANS REDIS
    # --------------------------------------------------------

    if redis_client is not None:
        try:
            # On stocke seulement les informations utiles.
            # La latence sera recalculée lors d'un HIT.
            cached_data = {
                "categorie": predicted_category,
                "score": round(confidence_score, 4),
            }

            redis_client.setex(
                cache_key,
                CACHE_TTL,
                json.dumps(cached_data),
            )

        except Exception as error:
            print(f"Erreur Redis SET : {error}")

    return response_data


# ============================================================
# GET /metrics
# ============================================================

@app.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )