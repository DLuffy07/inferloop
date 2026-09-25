"""Évalue le modèle zero-shot sur articles_test.csv, exactement comme l'API l'appelle.

Usage (depuis la racine du repo, avec requirements-eval.txt installé) :
    python scripts/evaluate_model.py                     # compare les modes "simple" et "fr"
    python scripts/evaluate_model.py --modes fr --quantize
    python scripts/evaluate_model.py --modes fr --write-baseline   # met à jour model/drift_baseline.json

Sorties :
    reports/evaluation_<mode>[_q8].csv   prédictions + latence par article
    reports/model_evaluation.md           tableau comparatif (accuracy, F1, latences)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.labels import CATEGORIES, build_input, get_prompt  # noqa: E402

DEFAULT_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"


def load_pipeline(model_name: str, quantize: bool):
    import torch
    from transformers import pipeline

    clf = pipeline("zero-shot-classification", model=model_name, tokenizer=model_name)
    if quantize:
        clf.model = torch.quantization.quantize_dynamic(clf.model, {torch.nn.Linear}, dtype=torch.qint8)
    return clf


def evaluate(clf, df: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, dict]:
    template, candidates = get_prompt(mode)
    labels = list(candidates)
    clf("Article de test.", candidate_labels=labels, hypothesis_template=template)  # warm-up

    preds, latencies = [], []
    for i, row in enumerate(df.itertuples(), start=1):
        text = build_input(row.texte, row.titre)
        start = time.perf_counter()
        out = clf(text, candidate_labels=labels, hypothesis_template=template, batch_size=len(labels))
        latencies.append((time.perf_counter() - start) * 1000)
        preds.append(candidates[out["labels"][0]])
        if i % 50 == 0:
            print(f"  [{mode}] {i}/{len(df)}")

    res = df.copy()
    res["prediction"] = preds
    res["latence_ms"] = latencies
    lat = np.array(latencies)
    metrics = {
        "mode": mode,
        "accuracy": accuracy_score(res.categorie, res.prediction),
        "f1_macro": f1_score(res.categorie, res.prediction, average="macro"),
        "p50": np.percentile(lat, 50),
        "p95": np.percentile(lat, 95),
        "p99": np.percentile(lat, 99),
        "report": classification_report(res.categorie, res.prediction, labels=CATEGORIES, zero_division=0),
        "confusion": pd.DataFrame(
            confusion_matrix(res.categorie, res.prediction, labels=CATEGORIES), index=CATEGORIES, columns=CATEGORIES
        ),
        "shares": res.prediction.value_counts(normalize=True).reindex(CATEGORIES, fill_value=0).round(4).to_dict(),
    }
    return res, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--modes", nargs="+", default=["simple", "fr"])
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="N premiers articles (debug)")
    parser.add_argument("--write-baseline", action="store_true", help="baseline de drift = dernier mode évalué")
    args = parser.parse_args()

    df = pd.read_csv(ROOT / "articles_test" / "articles_test.csv")
    df["texte"] = df["texte"].fillna("")  # 25 articles n'ont qu'un titre
    if args.limit:
        df = df.head(args.limit)
    print(f"{len(df)} articles — modèle {args.model} — quantize={args.quantize}")

    clf = load_pipeline(args.model, args.quantize)
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    suffix = "_q8" if args.quantize else ""

    all_metrics = []
    for mode in args.modes:
        res, m = evaluate(clf, df, mode)
        res.to_csv(reports / f"evaluation_{mode}{suffix}.csv", index=False)
        all_metrics.append(m)
        print(f"\n=== mode {mode} ===\naccuracy {m['accuracy']:.4f} | F1 macro {m['f1_macro']:.4f} | "
              f"p50 {m['p50']:.0f} ms | p95 {m['p95']:.0f} ms | p99 {m['p99']:.0f} ms")
        print(m["report"])
        print(m["confusion"])

    lines = [
        "# Évaluation du modèle sur articles_test.csv",
        "",
        f"Modèle : `{args.model}` — quantization int8 : {args.quantize} — {len(df)} articles — "
        f"généré le {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        "Latence = modèle seul, CPU local, 1 requête à la fois.",
        "",
        "| Mode prompt | Accuracy | F1 macro | p50 (ms) | p95 (ms) | p99 (ms) |",
        "|---|---|---|---|---|---|",
    ]
    for m in all_metrics:
        lines.append(f"| {m['mode']} | {m['accuracy']:.3f} | {m['f1_macro']:.3f} | "
                     f"{m['p50']:.0f} | {m['p95']:.0f} | {m['p99']:.0f} |")
    for m in all_metrics:
        lines += ["", f"## Mode `{m['mode']}`", "", "```", m["report"], "```", "",
                  "Matrice de confusion (lignes = réel, colonnes = prédit) :", "", "```",
                  m["confusion"].to_string(), "```"]
    (reports / f"model_evaluation{suffix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nRapport : reports/model_evaluation{suffix}.md")

    if args.write_baseline:
        last = all_metrics[-1]
        baseline = {
            "description": "Distribution des classes prédites sur articles_test.csv — référence du suivi de drift.",
            "source": f"reports/evaluation_{last['mode']}{suffix}.csv",
            "model": args.model,
            "prompt_mode": last["mode"],
            "quantized": args.quantize,
            "n_articles": len(df),
            "shares": last["shares"],
        }
        path = ROOT / "model" / "drift_baseline.json"
        path.write_text(json.dumps(baseline, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Baseline de drift mise à jour : {path}")


if __name__ == "__main__":
    main()
