import time
import pandas as pd
import numpy as np

from transformers import pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
CSV_PATH = "articles_test/articles_test.csv"

LABELS = [
    "politique",
    "économie",
    "sport",
    "culture",
    "faits divers"
]

LABEL_MAPPING = {
    "politique": "politique",
    "économie": "economie",
    "sport": "sport",
    "culture": "culture",
    "faits divers": "faits_divers"
}

# ============================================================
# CHARGEMENT DES DONNEES
# ============================================================

print("Chargement du dataset...")

df = pd.read_csv(CSV_PATH)

print(f"Nombre d'articles : {len(df)}")
print("\nRépartition des catégories :")
print(df["categorie"].value_counts())

# ============================================================
# CHARGEMENT DU MODELE
# ============================================================

print("\nChargement du modèle...")
print(f"Modèle : {MODEL_NAME}")

classifier = pipeline(
    "zero-shot-classification",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME
)

print("Modèle chargé.")

# ============================================================
# WARM-UP
# ============================================================

print("\nWarm-up du modèle...")

classifier(
    "Article de test pour initialiser le modèle.",
    candidate_labels=LABELS
)

print("Warm-up terminé.")

# ============================================================
# EVALUATION
# ============================================================

predictions = []
latencies = []

print("\nDébut de l'évaluation...\n")

for index, row in df.iterrows():

    # On utilise le titre + le corps de l'article
    text = f"{row['titre']}. {row['texte']}"

    start = time.perf_counter()

    result = classifier(
        text,
        candidate_labels=LABELS
    )

    latency_ms = (time.perf_counter() - start) * 1000

    prediction_model = result["labels"][0]
    prediction = LABEL_MAPPING[prediction_model]

    predictions.append(prediction)
    latencies.append(latency_ms)

    print(
        f"[{index + 1:03d}/{len(df)}] "
        f"Réel: {row['categorie']:<15} "
        f"Prédit: {prediction:<15} "
        f"Latence: {latency_ms:.0f} ms"
    )

# ============================================================
# METRIQUES ML
# ============================================================

y_true = df["categorie"].tolist()
y_pred = predictions

accuracy = accuracy_score(y_true, y_pred)

precision, recall, f1, _ = precision_recall_fscore_support(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

print("\n" + "=" * 60)
print("RESULTATS DU MODELE")
print("=" * 60)

print(f"\nAccuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")

print("\nRapport par catégorie :\n")

print(
    classification_report(
        y_true,
        y_pred,
        labels=[
        "politique",
        "economie",
        "sport",
        "culture",
        "faits_divers"
    ],
        zero_division=0
    )
)

# ============================================================
# MATRICE DE CONFUSION
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=LABELS
)

DATASET_LABELS = [
    "politique",
    "economie",
    "sport",
    "culture",
    "faits_divers"
]

cm_df = pd.DataFrame(
    cm,
    index=DATASET_LABELS,
    columns=DATASET_LABELS
)

print("\nMatrice de confusion :\n")
print(cm_df)

# ============================================================
# LATENCES
# ============================================================

latencies = np.array(latencies)

print("\n" + "=" * 60)
print("LATENCES")
print("=" * 60)

print(f"Moyenne : {np.mean(latencies):.0f} ms")
print(f"Médiane : {np.median(latencies):.0f} ms")
print(f"P50     : {np.percentile(latencies, 50):.0f} ms")
print(f"P95     : {np.percentile(latencies, 95):.0f} ms")
print(f"P99     : {np.percentile(latencies, 99):.0f} ms")
print(f"Min     : {np.min(latencies):.0f} ms")
print(f"Max     : {np.max(latencies):.0f} ms")

# ============================================================
# SAUVEGARDE
# ============================================================

df["prediction"] = predictions
df["latence_ms"] = latencies

df.to_csv(
    "evaluation_results.csv",
    index=False
)

print("\nRésultats détaillés sauvegardés dans evaluation_results.csv")