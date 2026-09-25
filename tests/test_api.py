"""Tests unitaires de l'API — le modèle et Redis sont mockés (pas de torch requis)."""

import json
import time
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from api import main
from api.labels import CATEGORIES, PROMPT_MODES, build_input, get_prompt

# TestClient sans context manager : le lifespan (chargement du vrai modèle) n'est pas exécuté.
client = TestClient(main.app)


def fake_pipeline_output(top: str = "sport"):
    """Sortie au format HuggingFace zero-shot, avec les libellés du mode courant."""
    labels_by_cat = {cat: text for text, cat in main.CANDIDATES.items()}
    order = [top] + [c for c in CATEGORIES if c != top]
    scores = [0.90, 0.04, 0.03, 0.02, 0.01]
    return {"sequence": "...", "labels": [labels_by_cat[c] for c in order], "scores": scores}


class FakeRedis:
    def __init__(self):
        self.store = {}
        self.ttl = {}

    def ping(self):
        return True

    def get(self, key):
        return self.store.get(key)

    def setex(self, key, ttl, value):
        self.store[key] = value
        self.ttl[key] = ttl


@pytest.fixture
def model(monkeypatch):
    clf = MagicMock(return_value=fake_pipeline_output("sport"))
    monkeypatch.setattr(main, "classifier", clf)
    return clf


@pytest.fixture
def no_model(monkeypatch):
    monkeypatch.setattr(main, "classifier", None)


@pytest.fixture
def cache(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(main, "redis_client", fake)
    return fake


@pytest.fixture
def no_cache(monkeypatch):
    monkeypatch.setattr(main, "redis_client", None)


# ------------------------------------------------------------------ /health

def test_health_ok(model, cache):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["redis_connected"] is True
    assert data["model"] == main.MODEL_NAME


def test_health_model_not_loaded(no_model, no_cache):
    r = client.get("/health")
    assert r.status_code == 503
    assert r.json()["model_loaded"] is False
    assert r.json()["redis_connected"] is False


def test_health_redis_down(model, monkeypatch):
    broken = MagicMock()
    broken.ping.side_effect = ConnectionError("down")
    monkeypatch.setattr(main, "redis_client", broken)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["redis_connected"] is False


# ------------------------------------------------------------------ /infer

def test_infer_returns_category(model, no_cache):
    r = client.post("/infer", json={"texte": "Le PSG remporte son match."})
    assert r.status_code == 200
    data = r.json()
    assert data["categorie"] == "sport"
    assert data["score"] == 0.90
    assert set(data["scores"]) == set(CATEGORIES)
    assert data["cache"] == "MISS"
    assert data["latence_ms"] >= 0


def test_infer_concatenates_title(model, no_cache):
    client.post("/infer", json={"titre": "Ligue 1", "texte": "Le PSG gagne."})
    sent_text = model.call_args.args[0]
    assert sent_text == "Ligue 1. Le PSG gagne."
    kwargs = model.call_args.kwargs
    assert kwargs["hypothesis_template"] == main.TEMPLATE
    assert kwargs["batch_size"] == len(CATEGORIES)


def test_infer_cache_miss_then_hit(model, cache):
    payload = {"texte": "Un article identique."}
    first = client.post("/infer", json=payload).json()
    second = client.post("/infer", json=payload).json()

    assert first["cache"] == "MISS"
    assert second["cache"] == "HIT"
    assert second["categorie"] == first["categorie"]
    assert model.call_count == 1  # le 2e appel n'a pas touché le modèle
    key = main.get_cache_key("Un article identique.")
    assert cache.ttl[key] == main.CACHE_TTL
    assert "latence_ms" not in json.loads(cache.store[key])


def test_infer_title_only(model, no_cache):
    # 25 articles de articles_test.csv n'ont qu'un titre
    r = client.post("/infer", json={"titre": "Roland-Garros : Djokovic s'offre un 24e titre"})
    assert r.status_code == 200
    assert model.call_args.args[0] == "Roland-Garros : Djokovic s'offre un 24e titre"


def test_infer_redis_errors_are_not_fatal(model, monkeypatch):
    broken = MagicMock()
    broken.get.side_effect = ConnectionError("down")
    broken.setex.side_effect = ConnectionError("down")
    monkeypatch.setattr(main, "redis_client", broken)
    r = client.post("/infer", json={"texte": "Redis est tombé."})
    assert r.status_code == 200
    assert r.json()["cache"] == "MISS"


@pytest.mark.parametrize(
    "payload",
    [
        {"texte": ""},
        {"texte": "   ", "titre": "  "},
        {"texte": "a" * 5001},
        {},
        {"texte": 123},
        {"titre": "t" * 501, "texte": "ok"},
    ],
)
def test_infer_invalid_input_422(model, payload):
    assert client.post("/infer", json=payload).status_code == 422


def test_infer_model_unavailable_503(no_model, no_cache):
    r = client.post("/infer", json={"texte": "Article de test"})
    assert r.status_code == 503


def test_infer_model_error_500(model, no_cache):
    model.side_effect = RuntimeError("boom")
    r = client.post("/infer", json={"texte": "Article"})
    assert r.status_code == 500


def test_infer_timeout_504(monkeypatch, no_cache):
    def slow(*_args, **_kwargs):
        time.sleep(0.3)
        return fake_pipeline_output()

    monkeypatch.setattr(main, "classifier", slow)
    monkeypatch.setattr(main, "INFERENCE_TIMEOUT", 0.05)
    r = client.post("/infer", json={"texte": "Article lent"})
    assert r.status_code == 504


def test_cache_key_is_versioned_and_deterministic():
    k1 = main.get_cache_key("abc")
    assert k1 == main.get_cache_key("abc")
    assert k1 != main.get_cache_key("abd")
    assert k1.startswith(f"inferloop:{main.MODEL_NAME}:{main.PROMPT_MODE}:")


# ------------------------------------------------------------------ /feedback

def test_feedback_disagreement(tmp_path, monkeypatch):
    path = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(main, "FEEDBACK_PATH", path)
    r = client.post(
        "/feedback",
        json={"texte": "Procès d'un élu", "categorie_predite": "faits_divers", "categorie_correcte": "politique"},
    )
    assert r.status_code == 201
    assert r.json()["accord"] == "non"
    record = json.loads(path.read_text(encoding="utf-8").strip())
    assert record["categorie_correcte"] == "politique"
    assert len(record["text_sha256"]) == 64


def test_feedback_agreement_and_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "FEEDBACK_PATH", tmp_path / "f.jsonl")
    ok = client.post("/feedback", json={"texte": "x", "categorie_predite": "sport", "categorie_correcte": "sport"})
    unk = client.post("/feedback", json={"texte": "x", "categorie_correcte": "sport"})
    assert ok.json()["accord"] == "oui"
    assert unk.json()["accord"] == "inconnu"


def test_feedback_invalid_category_422():
    r = client.post("/feedback", json={"texte": "x", "categorie_correcte": "meteo"})
    assert r.status_code == 422


# ------------------------------------------------------------------ /metrics

def test_metrics_exposes_business_and_http_metrics(model, no_cache):
    client.post("/infer", json={"texte": "Match de rugby."})
    client.post("/infer", json={"texte": ""})
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.text
    for name in (
        "inferloop_inference_total",
        "inferloop_http_request_duration_seconds_bucket",
        "inferloop_cache_hit_total",
        "inferloop_prediction_total",
        "inferloop_drift_baseline_share",
    ):
        assert name in body
    assert 'path="/infer",status="422"' in body
    assert 'path="/infer",status="200"' in body


def test_drift_baseline_loaded():
    shares = main.load_drift_baseline()
    assert set(shares) == set(CATEGORIES)
    assert abs(sum(shares.values()) - 1) < 1e-6


def test_drift_baseline_missing_file(tmp_path):
    assert main.load_drift_baseline(tmp_path / "absent.json") == {}


def test_root():
    assert client.get("/").json()["service"] == "InferLoop"


# ------------------------------------------------------------------ labels

def test_prompt_modes_cover_all_categories():
    for mode in PROMPT_MODES:
        _template, candidates = get_prompt(mode)
        assert sorted(candidates.values()) == sorted(CATEGORIES)


def test_unknown_prompt_mode():
    with pytest.raises(ValueError):
        get_prompt("inexistant")


def test_build_input():
    assert build_input(" corps ", " titre ") == "titre. corps"
    assert build_input("corps") == "corps"
    assert build_input("", "titre seul") == "titre seul"
    assert build_input(None, None) == ""
