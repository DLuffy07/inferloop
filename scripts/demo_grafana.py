"""Scénario scripté (~6 min) pour faire vivre le dashboard Grafana et déclencher les 4 alertes.

    python scripts/demo_grafana.py            # scénario complet, avec crash de l'API
    python scripts/demo_grafana.py --no-crash # sans l'étape de crash

Avant de lancer : Grafana ouvert sur http://localhost:3000 (dashboard InferLoop, période « Last 15 minutes »),
un 2e onglet sur Alerting > Alert rules. Au début de chaque phase, le script explique (👀) ce que Grafana montre.

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


def explain(*lines: str):
    """Affiche ce qu'on peut observer dans Grafana à cette étape, et pourquoi."""
    print("\n   👀 Dans Grafana, on peut observer :")
    for line in lines:
        # une ligne qui commence par deux espaces prolonge le point précédent
        print(f"        {line.strip()}" if line.startswith("  ") else f"      • {line}")
    print()


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
    explain(
        "« Réplicas API up » à 1 et « Taux d'erreur » proche de 0 % : le service est sain.",
        "« Latence /infer » : p50 autour de 0,5 s, le temps d'une inférence du modèle sur CPU ;",
        "  les requêtes servies depuis Redis tombent à quelques ms, d'où l'écart entre p50 et p99.",
        "« Cache hit rate » : ~30 % des articles sont déjà vus, ils ne passent pas par le modèle.",
        "« Écart par classe vs baseline » : toutes les barres restent vertes, proches de 0,",
        "  la répartition des prédictions ressemble à celle mesurée sur les 500 articles de test.",
    )
    run_phase(url, 90, workers=1, pool=ROWS, pause=0.3, hot_ratio=0.3, scale=s)

    banner("PHASE 2/5 — Pic de charge : 25 clients simultanés (75 s)")
    explain(
        "« File d'inférence & délestage » : la file monte à son maximum (2 en cours + 4 en attente)",
        "  et des rejets 503 apparaissent : l'API refuse tout de suite ce qu'elle ne peut pas traiter.",
        "« Requêtes/s par code HTTP » : les 200 plafonnent vers 2,4/s (capacité CPU du modèle),",
        "  tout le surplus part en 503 en quelques ms au lieu d'attendre un timeout.",
        "« Latence /infer » : le p99 grimpe à plusieurs secondes, c'est le temps d'attente dans la file.",
        "Alerting : « Latence p99 > 500 ms » et « Taux d'erreur > 5 % » passent en Pending puis Firing",
        "  (après 30 s de dépassement continu, pour éviter les fausses alertes sur un pic isolé).",
        "« Réplicas API up » reste à 1 : l'API sature mais ne tombe pas.",
    )
    run_phase(url, 75, workers=25, pool=ROWS, pause=0.3, hot_ratio=0.0, scale=s)

    banner("PHASE 3/5 — Drift : uniquement des articles sport (75 s)")
    explain(
        "« Part de chaque classe prédite » : la courbe sport monte nettement au-dessus des autres.",
        "« Écart par classe vs baseline » : la barre sport passe au rouge (> +20 points),",
        "  les autres classes deviennent négatives puisque leur part diminue d'autant.",
        "« Drift max vs baseline » dépasse 20 %, le seuil de l'alerte.",
        "Alerting : « Drift des prédictions > 20 points » passe en Firing.",
        "  En production, ce signal indiquerait un changement de l'actualité ou un modèle qui dérive :",
        "  on vérifierait alors un échantillon d'articles via POST /feedback avant de ré-entraîner.",
    )
    run_phase(url, 75, workers=2, pool=SPORT, pause=0.2, hot_ratio=0.0, scale=s)

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
        explain(
            "« Réplicas API up » passe à 0 le temps du redémarrage, puis revient à 1 :",
            "  Docker relance le conteneur tout seul grâce à la restart policy `unless-stopped`.",
            "Alerting : « API InferLoop down » passe en Firing pendant la coupure, puis repasse en Normal.",
            f"Le service est revenu en {time.time() - down_at:.0f} s sans intervention humaine",
            "  (le modèle est déjà dans le volume hf-cache, pas besoin de le re-télécharger).",
        )

    banner("PHASE 5/5 — Retour au calme (60 s)")
    explain(
        "Les courbes reviennent à leur niveau de la phase 1 : plus de 503, latence de nouveau stable.",
        "Alerting : les alertes repassent une à une en Normal ; celle du drift en dernier,",
        "  car elle se calcule sur une fenêtre glissante de 10 min.",
        "Avec la période « Last 15 minutes », le dashboard montre tout le scénario d'un coup :",
        "  trafic normal, saturation, drift, crash, puis retour à la normale.",
    )
    run_phase(url, 60, workers=1, pool=ROWS, pause=0.5, hot_ratio=0.5, scale=s)

    banner(f"Terminé. Codes HTTP : {dict(stats)}")


if __name__ == "__main__":
    main()
