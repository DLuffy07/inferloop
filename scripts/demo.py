"""Scénarios de démo InferLoop — Python standard uniquement (Windows / macOS / Linux).

    python scripts/demo.py smoke        # 10 articles réels via nginx, affiche catégorie/score/latence/cache
    python scripts/demo.py cache        # même article 2 fois -> MISS puis HIT
    python scripts/demo.py ratelimit    # rafale de 150 requêtes sur :80 -> apparition des 429
    python scripts/demo.py errors       # 60 requêtes invalides -> alerte "Taux d'erreur > 5 %"
    python scripts/demo.py drift        # 60 articles "sport" -> alerte drift (> 20 points)
    python scripts/demo.py feedback     # envoie une correction humaine sur /feedback
    python scripts/demo.py resilience   # suit /health pendant qu'on tue l'API (voir README)

Options : --url http://localhost (nginx, défaut) ou http://localhost:8000 (API directe)
"""

import argparse
import csv
import json
import random
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "articles_test" / "articles_test.csv"


def call(url: str, path: str, payload: dict | None = None, timeout: float = 15) -> tuple[int, dict]:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url + path, data=data, method="POST" if data else "GET", headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read() or b"{}")
        except ValueError:
            body = {}
        return e.code, body
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        return 0, {"erreur": str(e)}


def articles(categorie: str | None = None) -> list[dict]:
    with CSV_PATH.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if categorie is None or r["categorie"] == categorie]


def smoke(url):
    rows = random.sample(articles(), 10)
    ok = 0
    for r in rows:
        status, body = call(url, "/infer", {"titre": r["titre"], "texte": r["texte"]})
        good = body.get("categorie") == r["categorie"]
        ok += good
        print(f"{status} | réel={r['categorie']:<12} prédit={body.get('categorie', '-'):<12} "
              f"score={body.get('score', 0):.3f} {body.get('latence_ms', 0):>7.1f} ms "
              f"{body.get('cache', '')} {'✓' if good else '✗'} | {r['titre'][:50]}")
    print(f"\n{ok}/10 corrects")


def cache(url):
    payload = {"titre": "Démo cache", "texte": f"Le XV de France s'impose face à l'Irlande. {uuid.uuid4().hex[:6]}"}
    for i in (1, 2):
        status, body = call(url, "/infer", payload)
        print(f"appel {i} : {status} cache={body.get('cache')} latence={body.get('latence_ms')} ms")


def ratelimit(url):
    counts = {}
    for _ in range(150):
        status, _ = call(url, "/health")
        counts[status] = counts.get(status, 0) + 1
    print("codes HTTP :", counts, "(429 attendus au-delà de ~100 req/min + burst 20)")


def errors(url):
    for i in range(60):
        call(url, "/infer", {"texte": ""})  # 422
        if i % 10 == 0:
            print(f"{i}/60 requêtes invalides envoyées")
        time.sleep(0.7)
    print("Terminé — l'alerte 'Taux d'erreur /infer > 5 %' passe en Firing sous ~1 min.")


def drift(url):
    rows = articles("sport")
    for i in range(60):
        r = random.choice(rows)
        call(url, "/infer", {"titre": r["titre"], "texte": f"{r['texte']} ({uuid.uuid4().hex[:6]})"})
        if i % 10 == 0:
            print(f"{i}/60 articles sport envoyés")
        time.sleep(0.7)
    print("Terminé — la part 'sport' dépasse la baseline de > 20 points : alerte drift.")


def feedback(url):
    payload = {
        "titre": "Mise en examen d'un ministre",
        "texte": "Le ministre a été mis en examen pour prise illégale d'intérêts.",
        "categorie_predite": "faits_divers",
        "categorie_correcte": "politique",
        "commentaire": "Affaire politico-judiciaire (limite connue du modèle)",
    }
    print(call(url, "/feedback", payload))


def resilience(url):
    print('Dans un autre terminal (simule un crash du process) :')
    print('  docker compose exec api python -c "import os,signal; os.kill(1, signal.SIGTERM)"')
    print("Ctrl+C pour arrêter.\n")
    down_since = None
    while True:
        status, body = call(url, "/health", timeout=2)
        now = time.strftime("%H:%M:%S")
        if status == 200:
            if down_since:
                print(f"{now} ✅ API revenue après {time.time() - down_since:.0f} s")
                down_since = None
            else:
                print(f"{now} ✅ 200 model_loaded={body.get('model_loaded')}")
        else:
            down_since = down_since or time.time()
            print(f"{now} ❌ {status} (down depuis {time.time() - down_since:.0f} s)")
        time.sleep(1)


SCENARIOS = {f.__name__: f for f in (smoke, cache, ratelimit, errors, drift, feedback, resilience)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", choices=SCENARIOS)
    parser.add_argument("--url", default="http://localhost")
    args = parser.parse_args()
    try:
        SCENARIOS[args.scenario](args.url.rstrip("/"))
    except KeyboardInterrupt:
        sys.exit(0)
