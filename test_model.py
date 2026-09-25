from transformers import pipeline
import time

MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

print("Chargement du modèle...")
print("Modele :", MODEL)

classifier = pipeline(
    "zero-shot-classification",
    model=MODEL
)

titre = "Budget 2025 adopté en première lecture à l'Assemblée"

texte = """
Selon nos informations, après trois semaines de débats et
l'utilisation du 49-3, l'Assemblée a adopté le PLF 2025 visant
un déficit à 3,2 pourcent du PIB.
"""

article = titre + " " + texte

categories = [
    "politique",
    "économie",
    "sport",
    "culture",
    "faits divers"
]

# Échauffement : on ne mesure pas le premier passage
classifier(
    "Ceci est un test.",
    candidate_labels=categories
)

start = time.perf_counter()

resultat = classifier(
    article,
    candidate_labels=categories
)

latence = (time.perf_counter() - start) * 1000

print("\n--- RESULTAT ---")

for label, score in zip(resultat["labels"], resultat["scores"]):
    print(f"{label:15} : {score:.4f}")

print(f"\nPrediction : {resultat['labels'][0]}")
print(f"Latence    : {latence:.0f} ms")