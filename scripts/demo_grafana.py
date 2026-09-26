"""Scénario scripté (~6 min) pour faire vivre le dashboard Grafana et déclencher les 4 alertes.

    python scripts/demo_grafana.py            # scénario complet, avec crash de l'API
    python scripts/demo_grafana.py --no-crash # sans l'étape de crash

Avant de lancer : Grafana ouvert sur http://localhost:3000 (dashboard InferLoop, période « Last 15 minutes »),
un 2e onglet sur Alerting > Alert rules. Le script affiche 📸 aux moments où faire les captures.

Phases :
  1. Trafic normal    (90 s) : tout est vert, latence, cache hit rate, répartition proche de la baseline
  2. Pic de charge    (75 s) : file pleine, délestage 503 -> alertes « Latence p99 » et « Taux d'erreur »
  3. Drift            (75 s) : uniquement des articles sport -> alerte « Drift »
  4. Crash de l'API   (~20 s) : kill du process -> alerte « API down », restart automatique
  5. Retour au calme  (60 s) : les alertes repassent en Normal
"""

import argparse
import csv
import json
import random
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

with (ROOT / "articles_test" / "articles_test.csv").open(encoding="utf-8") as f:
    ROWS = list(csv.DictReader(f))
SPORT = [r for r in ROWS if r["categorie"] == "sport"]
HOT = ROWS[:20]

stats: Counter = Counter()
lock = threading.Lock()


def post(url: str, payload: dict) -> int:
    req = urllib.request.Request(
        url + "/infer", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return 0  # connexion refusée (API en cours de redémarrage)


def article(pool: list[dict], unique: bool = True) -> dict:
    r = random.choice(pool)
    texte = r["texte"] or ""
    if unique:
        texte = f"{texte} (réf. {uuid.uuid4().hex[:6]})"
    return {"titre": r["titre"], "texte": texte}


def worker(url: str, stop: threading.Event, pool: list[dict], pause: float, hot_ratio: float):
    while not stop.is_set():
        if random.random() < hot_ratio:
            payload = article(HOT, unique=False)
        else:
            payload = article(pool)
        code = post(url, payload)
        with lock:
            stats[code] += 1
        stop.wait(pause)


def run_phase(url, seconds, workers, pool, pause, hot_ratio, scale):
    stop = threading.Event()
    threads = [
        threading.Thread(target=worker, args=(url, stop, pool, pause, hot_ratio), daemon=True) for _ in range(workers)
    ]
    for t in threads:
        t.start()
    end = time.time() + seconds * scale
    while time.time() < end:
        time.sleep(min(10, max(0.1, end - time.time())))
        with lock:
            snapshot = dict(stats)
        print(f"   {time.strftime('%H:%M:%S')}  codes HTTP cumulés : {snapshot}")
    stop.set()
    for t in threads:
        t.join(timeout=20)


def banner(text: str):
    print(f"\n{'=' * 78}\n{time.strftime('%H:%M:%S')}  {text}\n{'=' * 78}")


def shot(text: str):
    print(f"\n   📸 CAPTURE : {text}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000", help="API directe (pas de rate limit nginx)")
    parser.add_argument("--no-crash", action="store_true")
    parser.add_argument("--scale", type=float, default=1.0, help="facteur sur les durées (tests)")
    args = parser.parse_args()
    url, s = args.url.rstrip("/"), args.scale

    try:
        with urllib.request.urlopen(url + "/health", timeout=5) as r:
            print("API :", json.loads(r.read()))
    except Exception as e:
        raise SystemExit(f"API injoignable sur {url} : {e}") from None

    banner("PHASE 1/5 — Trafic normal (90 s)")
    run_phase(url, 90, workers=1, pool=ROWS, pause=0.3, hot_ratio=0.3, scale=s)
    shot("dashboard complet : tout vert, latence p50/p95/p99, cache hit rate, répartition vs baseline")

    banner("PHASE 2/5 — Pic de charge : 25 clients simultanés (75 s)")
    run_phase(url, 75, workers=25, pool=ROWS, pause=0.3, hot_ratio=0.0, scale=s)
    shot("panneau « File d'inférence & délestage » + « Requêtes/s par code HTTP » (503 en rouge)")
    shot("Alerting > Alert rules : « Latence p99 » et « Taux d'erreur » en Firing (ou Pending)")

    banner("PHASE 3/5 — Drift : uniquement des articles sport (75 s)")
    run_phase(url, 75, workers=2, pool=SPORT, pause=0.2, hot_ratio=0.0, scale=s)
    shot("rangée « Drift des prédictions » : sport largement au-dessus de sa baseline, écart > 20 %")
    shot("Alerting : « Drift des prédictions > 20 points » en Firing")

    if not args.no_crash:
        banner("PHASE 4/5 — Crash du process API (restart automatique par Docker)")
        subprocess.run(
            ["docker", "compose", "exec", "-T", "api", "python", "-c", "import os,signal; os.kill(1, signal.SIGTERM)"],
            check=False,
        )
        down_at = time.time()
        time.sleep(2)
        while True:
            try:
                with urllib.request.urlopen(url + "/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            time.sleep(1)
        print(f"   API revenue après {time.time() - down_at:.0f} s")
        shot("tuile « Réplicas API up » (passée à 0) + Alerting : « API InferLoop down » en Firing")

    banner("PHASE 5/5 — Retour au calme (60 s)")
    run_phase(url, 60, workers=1, pool=ROWS, pause=0.5, hot_ratio=0.5, scale=s)
    shot("Alerting : les alertes repassent en Normal · dashboard sur « Last 15 minutes » montrant tout le scénario")

    banner(f"Terminé. Codes HTTP : {dict(stats)}")


if __name__ == "__main__":
    main()
