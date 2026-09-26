"""Évalue l'API EN SERVICE sur les 500 articles de test (Python standard uniquement).

Mesure ce que le client voit vraiment : qualité + latence de bout en bout de l'API.
Passe par l'API directe (:8000) pour ne pas être freiné par le rate limit nginx.

    python scripts/eval_api.py                   # mode courant de l'API
    python scripts/eval_api.py --limit 100       # rapide (~1 min)

Pour comparer les prompts : changer PROMPT_MODE dans .env (fr / simple),
`docker compose up -d api`, attendre healthy, relancer ce script.
Le cache est versionné par mode : pas de faux HIT entre deux modes.
"""

import argparse
import csv
import json
import statistics
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CATEGORIES = ["politique", "economie", "sport", "culture", "faits_divers"]


def post(url: str, payload: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, {}


def percentile(values: list[float], p: float) -> float:
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    base = args.url.rstrip("/")

    with urllib.request.urlopen(base + "/health", timeout=5) as r:
        health = json.loads(r.read())
    mode = health.get("prompt_mode", "?")
    quantized = bool(health.get("quantized"))
    tag = f"{mode}_q8" if quantized else mode
    print(f"API : mode de prompt = {mode}, quantized = {quantized}")

    with (ROOT / "articles_test" / "articles_test.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[: args.limit]

    y_true, y_pred, latencies, errors, hits = [], [], [], 0, 0
    start = time.time()
    for i, row in enumerate(rows, 1):
        status, body = post(base + "/infer", {"titre": row["titre"], "texte": row["texte"] or ""})
        if status != 200:
            errors += 1
            continue
        y_true.append(row["categorie"])
        y_pred.append(body["categorie"])
        if body["cache"] == "HIT":
            hits += 1  # articles en double dans le CSV : latence cache, exclue des stats modèle
        else:
            latencies.append(body["latence_ms"])
        if i % 50 == 0:
            acc = sum(t == p for t, p in zip(y_true, y_pred, strict=True)) / len(y_true)
            print(f"{i}/{len(rows)}  accuracy courante {acc:.3f}")

    n = len(y_true)
    acc = sum(t == p for t, p in zip(y_true, y_pred, strict=True)) / n
    lines = [
        f"# Évaluation via l'API — mode `{mode}`, quantized={quantized}",
        "",
        f"{n} articles · {errors} erreurs · {hits} cache HIT (doublons du CSV) · durée {time.time() - start:.0f} s",
        f"Modèle : `{health.get('model')}`",
        "",
        "| Accuracy | F1 macro | Latence p50 | p95 | p99 | max |",
        "|---|---|---|---|---|---|",
    ]
    report, f1s = [], []
    for c in CATEGORIES:
        tp = sum(t == c and p == c for t, p in zip(y_true, y_pred, strict=True))
        fp = sum(t != c and p == c for t, p in zip(y_true, y_pred, strict=True))
        fn = sum(t == c and p != c for t, p in zip(y_true, y_pred, strict=True))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f1s.append(f1)
        report.append(f"| {c} | {prec:.2f} | {rec:.2f} | {f1:.2f} |")
    if latencies:
        lat = (f"{percentile(latencies, 50):.0f} ms | {percentile(latencies, 95):.0f} ms | "
               f"{percentile(latencies, 99):.0f} ms | {max(latencies):.0f} ms")
    else:
        lat = "n/a | n/a | n/a | n/a"
        print("\n⚠️  Toutes les réponses viennent du cache : l'API n'a pas changé de configuration "
              "(vérifier .env puis `docker compose up -d api` et `curl localhost:8000/health`).")
    lines.append(f"| **{acc:.3f}** | **{statistics.mean(f1s):.3f}** | {lat} |")
    lines += ["", "| Classe | Précision | Rappel | F1 |", "|---|---|---|---|", *report]
    shares = Counter(y_pred)
    lines += ["", "Répartition des prédictions : " + ", ".join(f"{c} {shares[c] / n:.1%}" for c in CATEGORIES)]
    confusion = Counter(zip(y_true, y_pred, strict=True))
    lines += ["", "Matrice de confusion (lignes = réel, colonnes = prédit) :", "", "```",
              "réel \\ prédit".ljust(14) + "".join(c[:10].rjust(11) for c in CATEGORIES)]
    for t in CATEGORIES:
        lines.append(t.ljust(14) + "".join(str(confusion[(t, p)]).rjust(11) for p in CATEGORIES))
    lines.append("```")

    text = "\n".join(lines) + "\n"
    print("\n" + text)
    out = ROOT / "reports" / f"eval_api_{tag}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"Rapport : {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
