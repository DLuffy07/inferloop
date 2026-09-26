"""InferLoop — API d'inférence MédiaLens.

Endpoints :
    POST /infer     classification d'un article (cache Redis)
    POST /feedback  collecte des corrections humaines (bonus)
    GET  /health    état du modèle et de Redis (503 si modèle non chargé)
    GET  /metrics   exposition Prometheus
"""

import asyncio
import hashlib
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import redis
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field, model_validator

from api.labels import CATEGORIES, build_input, get_prompt

# ============================================================
# CONFIGURATION (variables d'environnement, cf. .env.example)
# ============================================================

MODEL_NAME = os.getenv("MODEL_NAME", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
PROMPT_MODE = os.getenv("PROMPT_MODE", "simple").strip()
LOAD_MODEL = os.getenv("LOAD_MODEL", "1") == "1"
QUANTIZE = os.getenv("QUANTIZE", "0").strip() == "1"  # ⚠️ mesuré : casse mDeBERTa (voir model/README.md)
TORCH_THREADS = int(os.getenv("TORCH_THREADS", "0"))
INFERENCE_TIMEOUT = float(os.getenv("INFERENCE_TIMEOUT", "10"))  # filet de sécurité : la file bornée limite l'attente
# Inférences exécutées en parallèle par process. 1 = torch utilise tous les cœurs pour une
# seule inférence à la fois (pas de sur-souscription CPU, latence minimale par requête).
MAX_CONCURRENT_INFERENCES = int(os.getenv("MAX_CONCURRENT_INFERENCES", "1"))
# Requêtes autorisées à attendre derrière. Au-delà : 503 immédiat (load shedding)
# plutôt qu'un timeout après 5-10 s qui consomme du CPU pour rien.
# Réglé pour que l'attente max (≈ (MAX_QUEUE + 1) x 0,5 s) reste bien sous INFERENCE_TIMEOUT :
# sinon on calcule des réponses que le client a déjà abandonnées (mesuré : 208 x 504 avec 8 / 5 s).
MAX_QUEUE = int(os.getenv("MAX_QUEUE", "4"))
MODEL_LOAD_RETRIES = int(os.getenv("MODEL_LOAD_RETRIES", "3"))
MODEL_LOAD_RETRY_DELAY = float(os.getenv("MODEL_LOAD_RETRY_DELAY", "10"))

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))

BASE_DIR = Path(__file__).resolve().parent.parent
DRIFT_BASELINE_PATH = Path(
    os.getenv("DRIFT_BASELINE_PATH", BASE_DIR / "model" / "drift_baseline.json")
)
FEEDBACK_PATH = Path(os.getenv("FEEDBACK_PATH", BASE_DIR / "data" / "feedback.jsonl"))

MAX_TEXT_CHARS = 5000
TEMPLATE, CANDIDATES = get_prompt(PROMPT_MODE)  # CANDIDATES : texte candidat -> id canonique

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("inferloop")


# ============================================================
# METRIQUES PROMETHEUS
# ============================================================

LATENCY_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0)

HTTP_REQUESTS = Counter(
    "inferloop_http_requests_total",
    "Requêtes HTTP par méthode, route et code retour",
    ["method", "path", "status"],
)
HTTP_LATENCY = Histogram(
    "inferloop_http_request_duration_seconds",
    "Latence HTTP de bout en bout (côté API)",
    ["path"],
    buckets=LATENCY_BUCKETS,
)
INFLIGHT = Gauge("inferloop_inflight_requests", "Requêtes en cours de traitement")

REQUEST_COUNT = Counter("inferloop_request_total", "Requêtes valides reçues sur /infer")
INFERENCE_COUNT = Counter("inferloop_inference_total", "Inférences exécutées par le modèle")
INFERENCE_LATENCY = Histogram(
    "inferloop_inference_duration_seconds",
    "Temps d'inférence du modèle (hors cache)",
    buckets=LATENCY_BUCKETS,
)
ERROR_COUNT = Counter("inferloop_error_total", "Erreurs d'inférence", ["reason"])

CACHE_HIT_COUNT = Counter("inferloop_cache_hit_total", "Résultats servis depuis Redis")
CACHE_MISS_COUNT = Counter("inferloop_cache_miss_total", "Résultats absents de Redis")

PREDICTION_COUNT = Counter(
    "inferloop_prediction_total",
    "Prédictions servies par catégorie (cache inclus) — base du suivi de drift",
    ["categorie"],
)
DRIFT_BASELINE = Gauge(
    "inferloop_drift_baseline_share",
    "Part de référence de chaque catégorie (500 articles de test)",
    ["categorie"],
)
FEEDBACK_COUNT = Counter(
    "inferloop_feedback_total",
    "Retours humains reçus sur /feedback",
    ["accord"],
)
INFERENCE_QUEUE = Gauge(
    "inferloop_inference_queue", "Inférences en cours + en attente (hors cache)"
)
SHED_COUNT = Counter(
    "inferloop_shed_total", "Requêtes rejetées en 503 car la file d'inférence est pleine"
)
MODEL_LOADED = Gauge("inferloop_model_loaded", "1 si le modèle est chargé")
MODEL_INFO = Gauge(
    "inferloop_model_info", "Modèle servi", ["model", "prompt_mode", "quantized"]
)

# Pré-initialisation des séries : sans ça, increase() rate le premier incrément.
for _cat in CATEGORIES:
    PREDICTION_COUNT.labels(categorie=_cat)
for _accord in ("oui", "non", "inconnu"):
    FEEDBACK_COUNT.labels(accord=_accord)
for _reason in ("model_unavailable", "timeout", "inference", "overloaded"):
    ERROR_COUNT.labels(reason=_reason)


def load_drift_baseline(path: Path = DRIFT_BASELINE_PATH) -> dict[str, float]:
    try:
        shares = json.loads(path.read_text(encoding="utf-8"))["shares"]
    except (OSError, KeyError, ValueError) as error:
        logger.warning("Baseline de drift indisponible (%s) : %s", path, error)
        return {}
    for cat, share in shares.items():
        DRIFT_BASELINE.labels(categorie=cat).set(share)
    return shares


# ============================================================
# MODELE & REDIS
# ============================================================

classifier = None
redis_client = None
# Pool dédié au modèle : les inférences ne saturent pas le threadpool de FastAPI (utilisé
# pour Redis), et une requête expirée encore en file est annulée avant de consommer du CPU.
_inference_pool = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_INFERENCES, thread_name_prefix="inference")
_inflight_inferences = 0  # modifié uniquement depuis la boucle asyncio : pas de verrou nécessaire
_feedback_lock = threading.Lock()


def load_classifier():
    """Charge le pipeline HuggingFace (import tardif : les tests n'ont pas besoin de torch)."""
    import torch
    from transformers import pipeline

    if TORCH_THREADS > 0:
        torch.set_num_threads(TORCH_THREADS)

    start = time.perf_counter()
    clf = pipeline("zero-shot-classification", model=MODEL_NAME, tokenizer=MODEL_NAME)

    if QUANTIZE:
        # Quantification dynamique int8 des couches Linear : ~2x plus rapide sur CPU.
        clf.model = torch.quantization.quantize_dynamic(
            clf.model, {torch.nn.Linear}, dtype=torch.qint8
        )

    # Warm-up : la première inférence est toujours lente (allocations, JIT).
    clf("Article de test.", candidate_labels=list(CANDIDATES), hypothesis_template=TEMPLATE)
    logger.info("Modèle %s chargé en %.1fs (quantized=%s)", MODEL_NAME, time.perf_counter() - start, QUANTIZE)
    return clf


def connect_redis():
    """Crée toujours le client : redis-py se reconnecte seul à chaque commande.

    Si Redis est indisponible au démarrage, l'API tourne sans cache puis
    le retrouve automatiquement dès qu'il revient (pas de client figé à None).
    """
    client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        decode_responses=True,
        socket_connect_timeout=1,
        socket_timeout=1,
        retry_on_timeout=False,
    )
    try:
        client.ping()
        logger.info("Connexion Redis OK (%s:%s)", REDIS_HOST, REDIS_PORT)
    except Exception as error:
        logger.warning("Redis indisponible pour l'instant, cache désactivé jusqu'à son retour : %s", error)
    return client


async def load_classifier_with_retry():
    """Retente le chargement (réseau/DNS instable au démarrage), puis abandonne.

    En cas d'échec définitif, on lève l'exception : uvicorn s'arrête, le conteneur
    sort et la restart policy Docker le relance. Mieux qu'une API vivante
    mais bloquée pour toujours en 503.
    """
    for attempt in range(1, MODEL_LOAD_RETRIES + 1):
        try:
            return await run_in_threadpool(load_classifier)
        except Exception as error:
            logger.error("Chargement du modèle : tentative %d/%d échouée : %s", attempt, MODEL_LOAD_RETRIES, error)
            if attempt == MODEL_LOAD_RETRIES:
                raise
            await asyncio.sleep(MODEL_LOAD_RETRY_DELAY * attempt)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global classifier, redis_client
    load_drift_baseline()
    MODEL_INFO.labels(model=MODEL_NAME, prompt_mode=PROMPT_MODE, quantized=str(QUANTIZE)).set(1)
    redis_client = connect_redis()
    if LOAD_MODEL:
        classifier = await load_classifier_with_retry()
    MODEL_LOADED.set(1 if classifier is not None else 0)
    yield


app = FastAPI(
    title="InferLoop API",
    description="API d'inférence NLP pour MédiaLens",
    version="1.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    INFLIGHT.inc()
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        route = request.scope.get("route")
        path = route.path if route is not None else "unmatched"
        HTTP_REQUESTS.labels(request.method, path, str(status)).inc()
        HTTP_LATENCY.labels(path).observe(time.perf_counter() - start)
        INFLIGHT.dec()


# ============================================================
# SCHEMAS
# ============================================================

Categorie = Literal["politique", "economie", "sport", "culture", "faits_divers"]


class InferRequest(BaseModel):
    titre: str | None = Field(None, max_length=500, description="Titre de l'article (optionnel)")
    texte: str = Field("", max_length=MAX_TEXT_CHARS, description="Corps de l'article")

    @model_validator(mode="after")
    def titre_ou_texte(self):
        # 5 % des articles de test n'ont qu'un titre : on accepte, mais jamais un contenu vide
        if not self.texte.strip() and not (self.titre or "").strip():
            raise ValueError("Il faut au moins un titre ou un texte non vide.")
        return self


class InferResponse(BaseModel):
    categorie: Categorie
    score: float
    scores: dict[str, float]
    latence_ms: float
    cache: Literal["HIT", "MISS"]
    modele: str


class FeedbackRequest(InferRequest):
    categorie_predite: Categorie | None = None
    categorie_correcte: Categorie
    commentaire: str | None = Field(None, max_length=1000)


# ============================================================
# OUTILS
# ============================================================

def get_cache_key(text: str) -> str:
    """Clé = modèle + mode de prompt + SHA-256 du texte.

    Le préfixe versionne le cache : changer de modèle ou de prompt invalide
    naturellement les anciennes entrées. SHA-256 rend les collisions négligeables.
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"inferloop:{MODEL_NAME}:{PROMPT_MODE}:{digest}"


def _redis_ok() -> bool:
    if redis_client is None:
        return False
    try:
        return bool(redis_client.ping())
    except Exception:
        return False


def _cache_get(key: str) -> dict | None:
    if redis_client is None:
        return None
    try:
        raw = redis_client.get(key)
        return json.loads(raw) if raw else None
    except Exception as error:
        logger.warning("Redis GET en échec : %s", error)
        return None


def _cache_set(key: str, value: dict) -> None:
    if redis_client is None:
        return
    try:
        redis_client.setex(key, CACHE_TTL, json.dumps(value))
    except Exception as error:
        logger.warning("Redis SET en échec : %s", error)


def _predict(text: str) -> dict:
    result = classifier(
        text,
        candidate_labels=list(CANDIDATES),
        hypothesis_template=TEMPLATE,
        batch_size=len(CANDIDATES),  # les 5 paires NLI en un seul forward
    )
    pairs = zip(result["labels"], result["scores"], strict=True)
    scores = {CANDIDATES[label]: round(float(s), 4) for label, s in pairs}
    top = CANDIDATES[result["labels"][0]]
    return {"categorie": top, "score": scores[top], "scores": scores}


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
def root():
    return {"service": "InferLoop", "docs": "/docs", "endpoints": ["/infer", "/feedback", "/health", "/metrics"]}


@app.get("/health")
def health():
    loaded = classifier is not None
    body = {
        "status": "healthy" if loaded else "unhealthy",
        "model_loaded": loaded,
        "redis_connected": _redis_ok(),
        "model": MODEL_NAME,
        "prompt_mode": PROMPT_MODE,
        "quantized": QUANTIZE,
        "version": app.version,
    }
    if not loaded:
        # 503 : le healthcheck Docker et nginx considèrent le service comme indisponible
        return Response(json.dumps(body), status_code=503, media_type="application/json")
    return body


@app.post("/infer", response_model=InferResponse)
async def infer(request: InferRequest):
    global _inflight_inferences
    REQUEST_COUNT.inc()
    start = time.perf_counter()

    if classifier is None:
        ERROR_COUNT.labels(reason="model_unavailable").inc()
        raise HTTPException(status_code=503, detail="Le modèle n'est pas chargé.")

    text = build_input(request.texte, request.titre)
    key = get_cache_key(text)

    cached = await run_in_threadpool(_cache_get, key)
    if cached is not None:
        CACHE_HIT_COUNT.inc()
        PREDICTION_COUNT.labels(categorie=cached["categorie"]).inc()
        return {
            **cached,
            "latence_ms": round((time.perf_counter() - start) * 1000, 2),
            "cache": "HIT",
            "modele": MODEL_NAME,
        }
    CACHE_MISS_COUNT.inc()

    if _inflight_inferences >= MAX_CONCURRENT_INFERENCES + MAX_QUEUE:
        SHED_COUNT.inc()
        ERROR_COUNT.labels(reason="overloaded").inc()
        raise HTTPException(
            status_code=503,
            detail="Service saturé, réessayez dans quelques secondes.",
            headers={"Retry-After": "2"},
        )

    _inflight_inferences += 1
    INFERENCE_QUEUE.set(_inflight_inferences)
    start_inference = time.perf_counter()
    try:
        loop = asyncio.get_running_loop()
        prediction = await asyncio.wait_for(
            loop.run_in_executor(_inference_pool, _predict, text), timeout=INFERENCE_TIMEOUT
        )
    except TimeoutError:
        ERROR_COUNT.labels(reason="timeout").inc()
        raise HTTPException(status_code=504, detail=f"Inférence > {INFERENCE_TIMEOUT}s.") from None
    except Exception:
        ERROR_COUNT.labels(reason="inference").inc()
        logger.exception("Erreur lors de l'inférence")
        raise HTTPException(status_code=500, detail="Erreur lors de l'inférence.") from None
    finally:
        _inflight_inferences -= 1
        INFERENCE_QUEUE.set(_inflight_inferences)

    INFERENCE_COUNT.inc()
    INFERENCE_LATENCY.observe(time.perf_counter() - start_inference)
    PREDICTION_COUNT.labels(categorie=prediction["categorie"]).inc()

    await run_in_threadpool(_cache_set, key, prediction)

    return {
        **prediction,
        "latence_ms": round((time.perf_counter() - start) * 1000, 2),
        "cache": "MISS",
        "modele": MODEL_NAME,
    }


@app.post("/feedback", status_code=201)
def feedback(request: FeedbackRequest):
    """Collecte une correction humaine pour un futur re-training (JSONL append-only)."""
    if request.categorie_predite is None:
        accord = "inconnu"
    else:
        accord = "oui" if request.categorie_predite == request.categorie_correcte else "non"

    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "model": MODEL_NAME,
        "prompt_mode": PROMPT_MODE,
        "text_sha256": hashlib.sha256(build_input(request.texte, request.titre).encode()).hexdigest(),
        "accord": accord,
        **request.model_dump(),
    }
    try:
        FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _feedback_lock, FEEDBACK_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as error:
        logger.error("Écriture du feedback impossible : %s", error)
        raise HTTPException(status_code=500, detail="Feedback non enregistré.") from None

    FEEDBACK_COUNT.labels(accord=accord).inc()
    return {"status": "enregistré", "accord": accord}


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
