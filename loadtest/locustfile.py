"""Test de charge InferLoop — 50 utilisateurs simultanés.

Interface web :   docker compose --profile loadtest up locust   -> http://localhost:8089
Headless + rapport HTML : ./scripts/loadtest.sh

Mix de trafic réaliste :
  - 70 % d'articles "nouveaux" (texte rendu unique -> cache MISS, vraie inférence)
  - 30 % d'articles déjà vus (texte identique -> cache HIT Redis)
"""

import csv
import os
import random
import uuid
from pathlib import Path

from locust import HttpUser, between, task

CSV_PATH = Path(os.getenv("ARTICLES_CSV", Path(__file__).parent.parent / "articles_test" / "articles_test.csv"))

FALLBACK = [
    ("Ligue 1", "Le PSG remporte son match de championnat après une victoire nette."),
    ("Taux directeurs", "La Banque centrale européenne annonce une nouvelle décision sur les taux."),
    ("Réforme", "Le gouvernement présente une nouvelle réforme devant le Parlement."),
    ("Cinéma", "Un nouveau film français sort cette semaine dans les salles."),
    ("Incendie", "Un incendie s'est déclaré cette nuit dans un immeuble du centre-ville."),
]


def load_articles() -> list[tuple[str, str]]:
    try:
        with CSV_PATH.open(encoding="utf-8") as f:
            return [(row["titre"], row["texte"] or "") for row in csv.DictReader(f)]
    except OSError:
        return FALLBACK


ARTICLES = load_articles()
HOT_SET = ARTICLES[:20]  # petit ensemble rejoué pour générer des cache hits


class InferLoopUser(HttpUser):
    wait_time = between(1, 2)

    @task(7)
    def infer_new_article(self):
        titre, texte = random.choice(ARTICLES)
        self.client.post(
            "/infer",
            json={"titre": titre, "texte": f"{texte} (réf. {uuid.uuid4().hex[:8]})"},
            name="/infer [nouveau]",
        )

    @task(3)
    def infer_known_article(self):
        titre, texte = random.choice(HOT_SET)
        self.client.post("/infer", json={"titre": titre, "texte": texte}, name="/infer [déjà vu]")
