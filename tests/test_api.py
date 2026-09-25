from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from api import main

client = TestClient(main.app)


def test_health():
    """Vérifie que l'endpoint /health répond correctement."""

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert "status" in data
    assert "model_loaded" in data
    assert "model" in data


def test_infer():
    """Vérifie /infer en utilisant un faux modèle."""

    fake_classifier = MagicMock()

    fake_classifier.return_value = {
        "labels": [
            "sport",
            "politique",
            "culture",
            "économie",
            "faits divers"
        ],
        "scores": [
            0.90,
            0.04,
            0.03,
            0.02,
            0.01
        ]
    }

    original_classifier = main.classifier
    main.classifier = fake_classifier

    try:
        response = client.post(
            "/infer",
            json={
                "texte": "Le PSG remporte son match."
            }
        )

        assert response.status_code == 200

        data = response.json()

        assert data["categorie"] == "sport"
        assert data["score"] == 0.90
        assert "latence_ms" in data

    finally:
        main.classifier = original_classifier


def test_infer_empty_text():
    """Vérifie qu'un texte vide provoque une erreur 422."""

    response = client.post(
        "/infer",
        json={
            "texte": ""
        }
    )

    assert response.status_code == 422


def test_infer_text_too_long():
    """Vérifie la limite de 5000 caractères."""

    response = client.post(
        "/infer",
        json={
            "texte": "a" * 5001
        }
    )

    assert response.status_code == 422


def test_model_unavailable():
    """Vérifie que l'API retourne 503 si le modèle est indisponible."""

    original_classifier = main.classifier
    main.classifier = None

    try:
        response = client.post(
            "/infer",
            json={
                "texte": "Article de test"
            }
        )

        assert response.status_code == 503

    finally:
        main.classifier = original_classifier


def test_metrics():
    """Vérifie que l'endpoint Prometheus fonctionne."""

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "inferloop_inference_total" in response.text